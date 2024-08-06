import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def download_MNIST():
    """
    Download MNIST dataset (if not downloaded), and torch datasets
    """
    
    transform = transforms.Compose([
                transforms.ToTensor(),  # Convert the images to PyTorch tensors
    ])

    # Download and load the MNIST dataset & save to <data> folder
    train_dataset  = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
    test_dataset   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def convert_data_to_01(dataset, threshold, kind):
    """
    Converts 0-255 data to 0/1 according to a threshold, saves the tensors.
    
    Args:
        dataset (torchvision.datasets): torchvision dataset
        threshold (int): the value below threshold are set to 0 and above/equal set to 1
        kind (str): {'train', 'test'}
    """
    
    # Make directory if does not exist
    folder_path = './data/MNIST_binary/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the tensors
    torch.save((dataset.data>=threshold).int(), folder_path + 'X_' + kind + '.pt')
    torch.save(dataset.targets, folder_path + 'Y_' + kind + '.pt')
    
    print(f"The binary data is stored at ::: {folder_path}")


def plot_digits(X, Y=None, n = 25, random=True):
    """
    Plots n random digits along with their labels

    Args:
        X (torch Tensot): Shape n x h x w
        Y (torch Tensot): Shape n x h x w
        n (int, optional): No of samples to be plotted. Defaults to 25.
    """
    
    total_samples = X.shape[0]  
    random_indices = torch.randint(0, total_samples, (n,))     # random n samples
    
    if random:
        X = X[random_indices]
    
    if Y is not None:
        Y = Y[random_indices]
    
    # Define the number of rows and columns for the grid
    cols = 5
    rows = int(n//cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    # Plot images and labels
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X[i], cmap='gray')
        
        if Y is not None:
            ax.set_title(f'Label: {Y[i].item()}')
        
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_original_vs_encoded(x, model):
    """
    Plots the original image (org_img) side-by-side to the dec(enc(org_img))
    
    Args:
        x (torch Tenssor): shape [28x28] 
        model : trained Autoencoder model
    """
    
    x1 = x.unsqueeze(dim=0) # Add a batch dimension
    
    # Check device, encode ---> decode, send back to cpu
    device = (next(model.parameters())).device
    x2 =  model.encoder(x1.to(device))
    x2 = model.decoder.predict(x2).reshape(28,28).to('cpu')
    
    # Create a figure with 2 subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first tensor
    axs[0].imshow(x.squeeze(), cmap='gray')
    axs[0].set_title('Original Image')

    # Display the second tensor
    axs[1].imshow(x2, cmap='gray')
    axs[1].set_title('Encoded-Decoded Image')

    # Remove axis ticks
    for ax in axs:
        ax.axis('off')

    # Show the plot
    plt.show()
