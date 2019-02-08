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
    def __init__(self,path_dataset, show_variables = False):
        self.data_train, self.data_valid, self.data_test = read_data(path_dataset,show_variables)
        self.n_chan = self.data_train["fields"].shape[1]
        self.data_train["n_samples"] = self.data_train["dipoles"].shape[0]
        self.data_valid["n_samples"] = self.data_valid["dipoles"].shape[0]
        self.data_test["n_samples"] = self.data_test["dipoles"].shape[0]
        
        
# Define our learner
class dipfit(nn.Module):
    def __init__(self,n_chan,p_dropout):
        super().__init__()
        self.sizes = [1024,512,256]
        
        self.fc1 = nn.Linear(n_chan, self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], self.sizes[2])
        self.fc4 = nn.Linear(self.sizes[2], 6)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')        
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')        
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')        
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')     
        
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)       
        return x    
    

def get_localization_error(predicted,actual): # numpy input
    return np.sqrt(np.sum((predicted[:,:3] - actual[:,:3])**2,axis=1)) # in cm


def weighted_mse(predicted, actual, weight_localization=0.5): # torch input
    if weight_localization < 0 or weight_localization > 1:
        raise Exception("weight_localization should be a value in the range [0,1]")
    weight_moment = 1 - weight_localization
    loc_error = torch.mean(torch.mean((predicted[:,:3] - actual[:,:3])**2, dim=1))
    mom_error = torch.mean(torch.mean((predicted[:,3:] - actual[:,3:])**2, dim=1))
    return torch.mean(weight_localization * loc_error + weight_moment * mom_error)


def train_epoch(model,data,batch_size,device,optimizer,weight_localization,
     min_location,max_location,min_moment,max_moment):
    train_loss = 0.0
    batch_time = []
    for x, y in get_batches(data,batch_size):  
        batch_start = time.time()
        # Move to GPU
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        # Set gradients to 0
        optimizer.zero_grad()
        # Get output
        output = model(x)
        # Get loss
        loss = weighted_mse(normalize_dipole(output,data,device,min_location,max_location,min_moment,max_moment), normalize_dipole(y,data,device,min_location,max_location,min_moment,max_moment), weight_localization)    
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


def valid_epoch(model,data,batch_size,weight_localization,device,
     min_location,max_location,min_moment,max_moment):
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        batch_time = []
        for x, y in get_batches(data,batch_size):     
            batch_start = time.time()
            # move to GPU
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            output = model(x)
            ## update the average validation loss
            loss = weighted_mse(normalize_dipole(output,data,device,min_location,max_location,min_moment,max_moment), normalize_dipole(y,data,device,min_location,max_location,min_moment,max_moment), weight_localization)   
            valid_loss += loss.item() * x.size(0)
            # Measure batch time
            batch_time.append(time.time() - batch_start)
    valid_loss = valid_loss / data["n_samples"] 
    batch_time = np.mean(batch_time)
    return valid_loss, batch_time


def train(model,dataset,n_epochs,batch_size,device,optimizer,weight_localization,path_state_dict,
     min_location,max_location,min_moment,max_moment):
    valid_loss_min = np.Inf 
    train_loss = []
    valid_loss = []
    mean_train_batch_time = []
    mean_valid_batch_time = []
    # Time everything
    time_start = time.time()  
    for epoch in range(1, n_epochs+1):
        time_start_epoch = time.time()  
        # Train 
        model, train_loss_epoch, train_batch_time = train_epoch(model,dataset.data_train,batch_size,device,optimizer,weight_localization,
     min_location,max_location,min_moment,max_moment)
        train_loss.append(train_loss_epoch) 
        mean_train_batch_time.append(train_batch_time)
        # Validate
        valid_loss_epoch, valid_batch_time = valid_epoch(model,dataset.data_valid,batch_size,weight_localization,device,
     min_location,max_location,min_moment,max_moment)
        valid_loss.append(valid_loss_epoch)  
        mean_valid_batch_time.append(valid_batch_time)
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
    print("{} epochs ready in {:.3f} seconds. Minimum validation loss: {:.6f}".format(
        n_epochs, (time.time() - time_start), valid_loss_min))
    print("Train batch time: {:.5f} ± {:.5f} seconds".format(
        np.mean(mean_train_batch_time), np.std(mean_train_batch_time)))
    print("Valid batch time: {:.5f} ± {:.5f} seconds".format(
        np.mean(mean_valid_batch_time), np.std(mean_valid_batch_time)))
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
        y = data["dipoles"][idx,:]
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
        "dipoles": np.array(file['dipoles_train'],dtype=np.float32).squeeze(),
        "fields": np.array(file['field_train'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)} 
        
        data_valid = {
        "dipoles": np.array(file['dipoles_valid'],dtype=np.float32).squeeze(),
        "fields": np.array(file['field_valid'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)}       
        
        data_test = {
        "dipoles": np.array(file['dipoles_test'],dtype=np.float32).squeeze(),
        "dipoles_estimated": np.array(file['estimated_dipoles_test'],dtype=np.float32).squeeze(),
        "fields": np.array(file['field_test'],dtype=np.float32),
        "max_location": np.array(file['max_location'],dtype=np.float32),
        "max_moment": np.array(file['max_moment'],dtype=np.float32),
        "min_location": np.array(file['min_location'],dtype=np.float32),
        "min_moment": np.array(file['min_moment'],dtype=np.float32),
        "n_runs": np.array(file['n_runs'],dtype=np.float32),
        "snr": np.array(file['snr'],dtype=np.float32)}              
    return data_train, data_valid, data_test

























