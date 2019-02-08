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


# Define the dataset class
class dataset_from_mat():
    def __init__(self,path_dataset):
        self.data_train, self.data_valid, self.data_test = read_data(path_dataset)
        self.n_chan = self.data_train["fields"].shape[1]
        self.data_train["n_samples"] = self.data_train["dipoles"].shape[0]
        self.data_valid["n_samples"] = self.data_valid["dipoles"].shape[0]
        self.data_test["n_samples"] = self.data_test["dipoles"].shape[0]
        
        
# Define our learner
class dipfit(nn.Module):
    def __init__(self,n_chan,p_dropout):
        super().__init__()
#         self.fc1 = nn.Linear(n_chan, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 6)
        self.fc1 = nn.Linear(n_chan, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)
        self.dropout = nn.Dropout(p_dropout)        
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)       
        return x    
    
    
def train_epoch(model,data,batch_size,device,optimizer,criterion,
     min_location,max_location,min_moment,max_moment):
    train_loss = 0.0
    for x, y in get_batches(data,batch_size):     
        # Move to GPU
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        # Set gradients to 0
        optimizer.zero_grad()
        # Get output
        output = model(x)
        # Get loss
        loss = criterion(normalize_dipole(output,data,device,min_location,max_location,min_moment,max_moment), normalize_dipole(y,data,device,min_location,max_location,min_moment,max_moment))
        train_loss += loss.item() * x.size(0)
        # Calculate gradients
        loss.backward()
        # Take step
        optimizer.step()
    train_loss = train_loss / data["n_samples"] 
    return model, train_loss


def valid_epoch(model,data,batch_size,criterion,device,
     min_location,max_location,min_moment,max_moment):
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in get_batches(data,batch_size):     
            # move to GPU
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            output = model(x)
            ## update the average validation loss
            loss = criterion(normalize_dipole(output,data,device,min_location,max_location,min_moment,max_moment), normalize_dipole(y,data,device,min_location,max_location,min_moment,max_moment))
            valid_loss += loss.item() * x.size(0)
    valid_loss = valid_loss / data["n_samples"] 
    return valid_loss


def train(model,dataset,n_epochs,batch_size,device,optimizer,criterion,path_state_dict,
     min_location,max_location,min_moment,max_moment):
    valid_loss_min = np.Inf 
    train_loss = []
    valid_loss = []
    # Time everything
    time_start = time.time()  
    for epoch in range(1, n_epochs+1):
        time_start_epoch = time.time()  
        # Train 
        model, train_loss_epoch = train_epoch(model,dataset.data_train,batch_size,device,optimizer,criterion,
     min_location,max_location,min_moment,max_moment)
        train_loss.append(train_loss_epoch) 
        # Validate
        valid_loss_epoch = valid_epoch(model,dataset.data_valid,batch_size,criterion,device,
     min_location,max_location,min_moment,max_moment)
        valid_loss.append(valid_loss_epoch)  
        # Save if validation loss is the lowest so far
        if valid_loss_epoch <= valid_loss_min:
            torch.save(model.state_dict(), path_state_dict)
            valid_loss_min = valid_loss_epoch         
        # Print epoch statistics
        print('Epoch {} done in {:.2f} seconds. \tTraining Loss: {:.9f} \tValidation Loss: {:.9f}'.format(
            epoch,             
            time.time() - time_start_epoch,
            train_loss_epoch,
            valid_loss_epoch
            ))         
    # Show final statistics    
    print(f"{n_epochs} epochs ready in {(time.time() - time_start):.3f} seconds. Minimum validation loss: {valid_loss_min:.3f}")
    # Load best config
    model.load_state_dict(torch.load(path_state_dict))
    return model, train_loss, valid_loss
    
    
def get_batches(data,batch_size):
    # Let's always get the same batches  
    n_splits = round(data["n_samples"] / batch_size)    
    # Prepare indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf.get_n_splits(data["fields"])
    # Return data
    for _, idx in kf.split(data["fields"]):
        x = data["fields"][idx,:]
        y = data["dipoles"][idx,:,0]
        yield x, y
        
        
def normalize_dipole(dipoles,data,device,
     min_location,max_location,min_moment,max_moment):
    out = torch.zeros(dipoles.size(0),6)
    out[:,:3] = (dipoles[:,:3] - min_location) / (max_location - min_location);
    out[:,3:] = (dipoles[:,3:] - min_moment) / (max_moment - min_moment);
    return out


def read_data(dataset,show_variables = False):
    # Print Variables
    with h5py.File(dataset, 'r') as file:
        if show_variables:
            print(list(file.keys()))            
        # Get separate train, validation, and test data
        data_train = {
        "dipoles": np.array(file['dipoles_train'],dtype=np.float32),
        "fields": np.array(file['field_train'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)}          
        data_valid = {
        "dipoles": np.array(file['dipoles_valid'],dtype=np.float32),
        "fields": np.array(file['field_valid'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)}           
        data_test = {
        "dipoles": np.array(file['dipoles_test'],dtype=np.float32),
        "fields": np.array(file['field_test'],dtype=np.float32),
        "err_loc": np.array(file['err_loc'],dtype=np.float32),
        "err_mom": np.array(file['err_mom'],dtype=np.float32),
        "fitting_error": np.array(file['fitting_error'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)}              
    return data_train, data_valid, data_test

























