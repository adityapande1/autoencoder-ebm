
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _save_ebm_checkpoint(model, ckpt_name='default'):
    
    folder_path = './saved_ckpts/ebm/' 
    # Make directory if does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    ckpt_path = folder_path + ckpt_name + '.pth'    
    torch.save(model.state_dict(), ckpt_path)
    model.ebm_ckpt_path = ckpt_path


def grad_wrt_x(model, X):
    """
    Calculates dmodel_dx in a batch-wise fashion

    Args:
        model (nn model): Torch neural net model
        X (torch.Tensor): Shape(batch_size, input_dim)

    Returns:
        dmodel_dx : Gradient of model wrt x  
    """
    device = next(model.parameters()).device    # The device model is on
    X_params = nn.Parameter(X).to(device)       # Make parameter of X, so that gradients can be computed
    X_params.retain_grad()                      # Something to do with non leaf tensor
    
    if X_params.grad is not None:   # Clear earlier gradients of X_params
        X_params.grad.zero_()
    
    out = model(X_params).sum()     # Pass through neural net, make scalar so .sum() is used
    out.backward()                  # Do backpropogation

    return X_params.grad.to('cpu')            # Return gradients


def langevin_MCMC(f_theta, input_dim, num_steps, eps, batch_size, interval_samples=False):
    """
    Generates a batch of samples to be used for training (Contrastive Divergence) of energy based method 
    
    Args:
        f_theta (nn model): Torch neural net model
        input_dim (int): the dimension of the input vector
        num_steps (int): No of steps to run the MCMC loop for
        eps (float): Number close to zero
        batch_size (int)

    Returns:
        X (torch.Tensor) : Shape [batch_size, input_dim], Samples from MCMC sampling
    """
    
    # Intialise X_not (initial samples)
    X = torch.randn((batch_size, input_dim))
    X_samples = []
    
    for t in range(num_steps):
        
        Z = torch.randn((batch_size, input_dim))        # Gausssian noise
        df_dx = grad_wrt_x(model=f_theta, X=X)          # Calculate grad of f_theta(x) wrt theta, 
        X =  X + eps*df_dx + np.sqrt(2*eps)*Z           # x_new = x_old + eps*df_dx + sqrt(2*eps)*z
        
        if interval_samples and (t%5)==0:
            X_samples.append(X.clone())
    
    if interval_samples:
        return X, X_samples

    return X


def train_ebm_epoch(energy_model, autoencoder, dataloader, device, 
                optimizer, mcmc_samples_per_datapoint=8):
    """
    Trains a single epoch of energy_model

    Args:
        energy_model (nn model): Also called F_theta(x)
        autoencoder (nn model): 
        dataloader (torch dataloader): usually train_dataloader
        device (cpu or cuda): torch.Device
        optimizer (torch optimizer)
        mcmc_samples_per_datapoint (int, optional): Defaults to 8. No of samples for training for single datapoint
    """
    
    total_loss = 0.0                                    # Total epoch loss

    for idx, batch in enumerate(dataloader):
        
        x = batch[0].to(device)                         # Send data to device
        with torch.no_grad():
            x = autoencoder.encoder(x)                  
        
        num_datapoints = len(x)                         # In case when batch_size is not same (last lot)

        # Sample (mcmc_samples_per_datapoint*num_datapoints) datapoints
        x_sample = langevin_MCMC(f_theta=energy_model, input_dim=autoencoder.encoded_dim,  
                            batch_size=mcmc_samples_per_datapoint*num_datapoints,
                            num_steps=10000, eps=.001)
    
        x_sample = x_sample.view(num_datapoints, mcmc_samples_per_datapoint,-1)
        x_sample = x_sample.to(device)

        # Negative Log Likelihood, Backprop & Gradient Descent       
        optimizer.zero_grad()                           # Clear gradients           
        nll = -(energy_model(x) - energy_model(x_sample).mean(dim=1)).mean()
        nll.backward()
        optimizer.step()

        # Add up total points
        total_loss += nll.item()*num_datapoints
        print(f"\tBATCH : {idx+1} |  TOTAL BATCH LOSS: {round(nll.item(),8)}") 
  
        
    print(f"\n\tEPOCH COMPLETE | TOTAL EPOCH LOSS {round(total_loss/len(dataloader.dataset),8)}\n" )
    

def evaluate_ebm_model(energy_model, autoencoder, X, device):
    """
    Returns mean energy of a given dataset X
    
    Args:
        energy_model (nn model): Base energy model
        autoencoder (nn model): Base autoencoder
        X (torch tensor): Shape [batch_size, 28, 28]
        device (cuda or cpu)

    Returns:
        mean_energy: The mean energy of the dataset
    """
    
    
    X = X.to(device)                                # Send data to device
    with torch.no_grad():                           # Encode in order to send through ebm
        X = autoencoder.encoder(X)     
    
    energy_model.eval()
    mean_energy = energy_model(X).mean().item()     # Take average 
    energy_model.train()
    
    return mean_energy
    

    
def train_ebm(energy_model, autoencoder, X_train, X_test, device, batch_size,
             mcmc_samples_per_datapoint=8, num_epochs=3, lr=3e-4):

    X_train = TensorDataset(X_train)
    train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)   # Create a Train DataLoader
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=lr)              # Optimizer
    
    best_energy = float('inf')    # initiaise to infinity 
    # Get the current date and time and refer to it as model_name
    model_name = 'ebm_' +  datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    print(f"\nMODEL NAME : {model_name}")
    
    energy_model.to(device)                            # send model to device
    energy_model.train()                               # set model to train mode
    for epoch in range(num_epochs):
        
        print(f"############# EPOCH {epoch + 1} STARTED TRAINING #############\n")
        
        # Train for a single epoch
        train_ebm_epoch(energy_model=energy_model, autoencoder=autoencoder, dataloader=train_dataloader, 
                        device=device, mcmc_samples_per_datapoint=mcmc_samples_per_datapoint, optimizer=optimizer)
        
        ######################## EVALUATION & CKPT  ######################## 
        
        if X_test is not None:                  # If test dataset is given
            
            test_energy = evaluate_ebm_model(energy_model=energy_model, autoencoder=autoencoder, X=X_test, device=device)
            
            if test_energy < best_energy:           # Only save the best model
                
                _save_ebm_checkpoint(model=energy_model, ckpt_name=model_name)
                print(f"\tBEST MODEL @ EPOCH : [{epoch+1}/{num_epochs}]")
                best_energy = test_energy
            
        else:                                  # If test dataset is not given, save after every epoch
            test_energy = None
            _save_ebm_checkpoint(model=energy_model, ckpt_name=model_name)
            
        #################################################################### 
            