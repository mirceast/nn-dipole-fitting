import h5py
import torch
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt    
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator
from glob import glob
from sklearn.model_selection import KFold
from tqdm import tqdm
import pylab as pl
from IPython import display
import pdb
import pickle

# We create necessary subfolders if not present
useful_paths = {"temp": "./temp/", "trained": "./Trained/", "analysis": "../Analysis/"}
# Make a folder for each of the paths above if the folder doesn't exist
[os.mkdir(useful_paths[each]) for each in list(useful_paths) if not os.path.isdir(useful_paths[each])];

# Define our learner
class dipfit(nn.Module):
    def __init__(self,sizes,batchnorm=False):
        # sizes is for example [64,1024,512,256,3] for 64 inputs, 1024 neurons in layer 1, 256 in 3, and 3 outputs
        super().__init__()
        self.sizes = sizes
        self.batchnorm = batchnorm
        
        self.hidden = nn.ModuleList()
        for k in range(len(sizes)-1):
            self.hidden.append(nn.Linear(sizes[k], sizes[k+1]))
            nn.init.kaiming_normal_(self.hidden[k].weight, nonlinearity='relu')  
        
        if self.batchnorm:
            self.bn = nn.ModuleList()
            for k in range(len(sizes)-1): # -1 because we don't want batchnorm on the output
                self.bn.append(nn.BatchNorm1d(sizes[k+1]))   
        
    def forward(self, x): 
        if self.batchnorm:
            for k in range(len(self.hidden) - 1): # -1 because we don't want ReLU on the final output
                x = F.relu(self.bn[k](self.hidden[k](x)))          
        else:
            for k in range(len(self.hidden) - 1): # -1 because we don't want ReLU on the final output
                x = F.relu(self.hidden[k](x))     
        x = self.hidden[-1](x)       
        return x   
    
    
def check_cuda(gpu_idx = 0):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    if use_cuda and gpu_idx is not "cpu":
        print('CUDA is available!  Training on GPU ...')
        device = torch.device(f"cuda:{gpu_idx}")
        print("Using",torch.cuda.get_device_name(device))
    else:
        print('CUDA is not available.  Training on CPU ...')
        device = torch.device("cpu")
    return device
    
               
def normalize_dipole(dipoles,data,device,min_location,max_location,min_moment,max_moment):
    return dipoles
    out = torch.zeros(dipoles.size(0),6)
    out[:,:3] = (dipoles[:,:3] - min_location) / (max_location - min_location);
    out[:,3:] = (dipoles[:,3:] - min_moment) / (max_moment - min_moment);
    return out


def get_localization_error(predicted,actual):
    return torch.sqrt(torch.sum((predicted[:,:3] - actual[:,:3])**2, dim=1))


def train_epoch(model,data,batch_size,device,optimizer):
    train_loss = 0.0
    batch_time = []
    model.train()
    for x, y in get_batches(data,batch_size):  
        batch_start = time.time()
        # Move to GPU
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        # Set gradients to 0
        optimizer.zero_grad()
        # Get output
        output = model(x)
        # Get loss
        loss = torch.mean(get_localization_error(output,y))  
        train_loss += loss.item() * x.size(0)
        # Calculate gradients
        loss.backward()
        # Take step
        optimizer.step()
        # Measure batch time
        batch_time.append(time.time() - batch_start)
    train_loss = train_loss / data["n_samples"] 
    batch_time = np.mean(batch_time)
    return model, train_loss, batch_time


def valid_epoch(model,data,batch_size,device):
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        batch_time = []
        for x, y in get_batches(data,batch_size):     
            batch_start = time.time()
            # move to GPU
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            output = model(x)
            # Get loss
            loss = torch.mean(get_localization_error(output,y))  
            valid_loss += loss.item() * x.size(0)
            # Measure batch time
            batch_time.append(time.time() - batch_start)
    valid_loss = valid_loss / data["n_samples"] 
    batch_time = np.mean(batch_time)
    return valid_loss, batch_time


def show_loss_one_model(train_loss, valid_loss, title = None):
    n_epochs = len(train_loss)
    epochs = np.arange(1,n_epochs+1,1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.cla()
    ax.plot(epochs, train_loss, label='Train')
    ax.plot(epochs, valid_loss, label='Valid')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error [cm]")
    ax.legend()
    plt.minorticks_on()
    if title is not None:
        plt.title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show() 
    display.clear_output(wait=True)        
    

def train_and_validate_epoch(model,data_train,data_valid,batch_size,device,optimizer):
    # Train 
    model, train_loss_epoch, train_batch_time = train_epoch(model,data_train,batch_size,device,optimizer)
    # Validate
    valid_loss_epoch, valid_batch_time = valid_epoch(model,data_valid,batch_size,device) 
    return model, train_loss_epoch, valid_loss_epoch, train_batch_time, valid_batch_time


def get_model_name(path_model,sizes,n_epochs):
    sizes_str = str(sizes).replace("["," ").replace("]"," ").replace(", ","-")[1:-1]
    return "{}size_{}_epochs_{:03d}.pt".format(path_model,sizes_str,n_epochs)

    
def train(model,data_train,data_valid,n_epochs,batch_size,device,optimizer,lr_scheduler = None,show_every = np.inf):
    valid_loss_min = np.Inf 
    train_loss = []
    valid_loss = []
    mean_train_batch_time = []
    mean_valid_batch_time = []
    path_state_dict = f"./temp/temp_state_dict_{str(int(np.abs(np.random.randn()) * 1e12))}.pt"
    # Time everything
    time_start = time.time()  
    try:
        for epoch in range(1, n_epochs+1):
            time_start_epoch = time.time()  
            # Train and validate (in one function in case we interrupt from keyboard)
            model, train_loss_epoch, valid_loss_epoch, train_batch_time, valid_batch_time = train_and_validate_epoch(
                                                                        model,data_train,data_valid,batch_size,device,optimizer)
            train_loss.append(train_loss_epoch)
            mean_train_batch_time.append(train_batch_time)
            valid_loss.append(valid_loss_epoch) 
            mean_valid_batch_time.append(valid_batch_time) 
            # Save if validation loss is the lowest so far
            if valid_loss_epoch <= valid_loss_min:
                torch.save(model.state_dict(), path_state_dict)
                valid_loss_min = valid_loss_epoch           
            # Adjust learning rate if we have a scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(valid_loss_epoch)
            if not epoch%show_every and epoch is not 1:
                show_loss_one_model(train_loss,valid_loss)
    except KeyboardInterrupt:
        pass
    # Show final statistics    
    print("{} epochs ready in {:.3f} seconds. Minimum validation loss: {:.6f}".format(
        epoch, (time.time() - time_start), valid_loss_min))
    print("Train batch time: {:.5f} ± {:.5f} seconds".format(
        np.mean(mean_train_batch_time), np.std(mean_train_batch_time)))
    print("Valid batch time: {:.5f} ± {:.5f} seconds".format(
        np.mean(mean_valid_batch_time), np.std(mean_valid_batch_time)))
    show_loss_one_model(train_loss,valid_loss)
    # Load best config
    model.load_state_dict(torch.load(path_state_dict))
    os.remove(path_state_dict)
    return model, train_loss, valid_loss
    
    
def get_batches(data,batch_size):
    # Let's always get the same batches  
    n_splits = round(data["n_samples"] / batch_size)    
    # Prepare indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf.get_n_splits(data["fields"])
    # Return data
    for _, idx in kf.split(data["fields"]):
        # Prepare some noise
        noise = np.random.randn(len(idx),data["fields"].shape[1])
        noise = noise / np.linalg.norm(noise,axis=1)[:,None]        
        x = (np.random.exponential(size=(len(idx),1),scale=5)).astype(np.float32) * data["fields"][idx,:] + noise.astype(np.float32)
        y = data["dipoles"][idx,:]
        yield x, y
        
        
def get_test_pred(data_test, device, model = None, pred_dipoles=None, snr=np.nan):
    # Test dipoles location - ground truth

    test_dipoles = torch.from_numpy(data_test["dipoles"]).to(device)    
    # Calculate test loss for the network
    if pred_dipoles is None:
        model.eval()
        pred_dipoles = model(torch.from_numpy(data_test["fields"]).to(device))
    else:
        pred_dipoles = torch.from_numpy(pred_dipoles).to(device)
    loc_error = get_localization_error(pred_dipoles,test_dipoles).detach().to("cpu").numpy() 
    dipoles = {
        "loc": data_test["dipoles"],
        "loc_est": pred_dipoles.to("cpu").detach().numpy(),
        "loc_err": loc_error,
        "snr": snr}
    return dipoles


def save_dict(data,save_name):
    pickle_out = open(save_name,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()  
    
    
def load_dict(path):
    pickle_in = open(path,"rb")
    return pickle.load(pickle_in)
    
    
def save_model(model,train_loss,valid_loss,save_name):
    data = {"sizes": model.sizes,
       "batchnorm": model.batchnorm,
       "state_dict": model.state_dict(),
       "train_loss": train_loss,
       "valid_loss": valid_loss}
    torch.save(data, save_name)


def load_model(save_name):
    data = torch.load(save_name)
    model = dipfit(data["sizes"], batchnorm=data["batchnorm"])
    model.load_state_dict(data["state_dict"])
    train_loss = data["train_loss"]
    valid_loss = data["valid_loss"]
    return model, train_loss, valid_loss


def read_train_data(dataset,show_variables = False):
    # Print Variables
    with h5py.File(dataset, 'r') as file:
        if show_variables:
            print(list(file.keys()))            
        # Get separate train, validation, and test data
        data_train = {
        "dipoles": np.array(file['dipoles_train'],dtype=np.float32).squeeze()[:,:3],
        "fields": np.array(file['field_train'],dtype=np.float32),
        "n_samples": np.array(file['field_train'],dtype=np.float32).shape[0],
        "n_chan": np.array(file['field_train'],dtype=np.float32).shape[1]}                    
        data_valid = {
        "dipoles": np.array(file['dipoles_valid'],dtype=np.float32).squeeze()[:,:3],
        "fields": np.array(file['field_valid'],dtype=np.float32),
        "n_samples": np.array(file['field_valid'],dtype=np.float32).shape[0],
        "n_chan": np.array(file['field_valid'],dtype=np.float32).shape[1]}  
    return data_train, data_valid

def read_test_data(dataset,show_variables = False):
    # Print Variables
    with h5py.File(dataset, 'r') as file:
        if show_variables:
            print(list(file.keys()))            
        # Get separate train, validation, and test data
        data_test = {
        "dipoles": np.array(file['dipoles_test'],dtype=np.float32).squeeze()[:,:3],
        "fields": np.array(file['field_test'],dtype=np.float32),
        "n_samples": np.array(file['field_test'],dtype=np.float32).shape[0],
        "n_chan": np.array(file['field_test'],dtype=np.float32).shape[1],
        "dipoles_estimated": np.array(file['estimated_dipoles_test'],dtype=np.float32).squeeze()[:,:3],
        "snr": np.array(file['snr'],dtype=np.float32).squeeze()}   
    return data_test












