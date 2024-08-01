import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def _save_autoencoder_checkpoint(model, ckpt_name='default'):
    
    folder_path = './saved_ckpts/autoencoder/' 
    # Make directory if does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    ckpt_path = folder_path + ckpt_name + '.pth'    
    torch.save(model.state_dict(), ckpt_path)
    model.ckpt_path = ckpt_path


def _evaluate_model(model, criterion, data_loader):
    
    device = next(model.parameters()).device    # The device model is on
    total_loss, total_samples = 0, 0
    
    model.eval()                            # set model to evaluation mode
    for x in data_loader:
    
        x = x.to(device)
        
        y_pred = model(x)                   # Forward Pass
        y_true = x.view(x.shape[0],-1)      # Keep the batch dimension, flatten all other
        loss = criterion(y_pred, y_true)    # MSE loss : mean
        
        total_loss += loss.item()*len(y_pred)      # remove the denominator term
        total_samples += len(y_pred) 

    return (total_loss/total_samples)
    

    
def train_autoencoder(model, X_train, X_test, lr, num_epochs, batch_size=64, device='cpu'):
    
    train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)   # Create a Train DataLoader
    test_dataloader  = DataLoader(X_test, batch_size=batch_size, shuffle=False)   # Create a Test DataLoader
    
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')    # initiaise to infinity 

    # Get the current date and time and refer to it as model_name
    model_name = 'autoencoder_' +  datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    print(f"\nMODEL NAME : {model_name}")
    
    model.to(device)                            # send model to device
    model.train()                               # set model to train mode
    for epoch in range(num_epochs):
        for x in train_dataloader:
            
            x = x.to(device)
            
            optimizer.zero_grad()               # Clear gradients
            y_pred = model(x)                   # Forward Pass
            y_true = x.view(x.shape[0],-1)      # Keep the batch dimension, flatten all other
            
            loss = criterion(y_pred, y_true)    # MSE loss
            
            loss.backward()                     # Backpropogation   
            optimizer.step()                    # Gradient Descent
    
        

        ######################## EVALUATION & CKPT  ######################## 
        
        if X_test is not None:                  # If test dataset is given
            
            test_loss = _evaluate_model(model=model, criterion=criterion, data_loader=test_dataloader)
            
            if test_loss < best_loss:           # Only save the best model
                
                _save_autoencoder_checkpoint(model=model, ckpt_name=model_name)
                print(f"BEST MODEL @ EPOCH : [{epoch+1}/{num_epochs}]")
                best_loss = test_loss
            
        else:
            test_loss = None
            _save_autoencoder_checkpoint(model=model, ckpt_name=model_name)
        
        #################################################################### 

            
        print(f"Epoch [{epoch+1}/{num_epochs}]  |  Train Loss: {loss.item():.6f}  |  Test Loss: {test_loss:.6f}")
        
        
