import argparse

import matplotlib.pyplot as plt
import torch

from models.autoencoder.autoencoder import Autoencoder
from models.autoencoder.autoencoder_utils import train_autoencoder
from models.ebm.energy_net import EnergyNet
from models.ebm.energy_net_utils import evaluate_ebm_model, langevin_MCMC, train_ebm
from utils import (
    convert_data_to_01,
    download_MNIST,
    plot_digits,
    plot_original_vs_encoded,
)


def main():
    
    parser = argparse.ArgumentParser(description="Parse model training parameters")
    
    parser.add_argument('--auto_hidden_dim', type=int, required=True, help='Dimension of the hidden layer of autoencoder')
    parser.add_argument('--encoded_dim', type=int, required=True, help='Dimension of the encoded representation in encoder')
    parser.add_argument('--auto_num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--auto_batch_size', type=int, required=True, help='Size of each training batch')
    parser.add_argument('--auto_lr', type=float, required=True, help='Learning rate for the optimizer')
    
    parser.add_argument('--ebm_hidden_dim', type=int, required=True, help='Dimension of the hidden layer of autoencoder')
    parser.add_argument('--ebm_num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--ebm_batch_size', type=int, required=True, help='Size of each training batch')
    parser.add_argument('--ebm_lr', type=float, required=True, help='Learning rate for the optimizer')
    parser.add_argument('--mcmc_samples_per_datapoint', type=int, required=True, help='Number of MCMC samples per datapoint')
    
    args = parser.parse_args()

    #### DATASET ############################################################
    
    # Download MNIST Dataset
    train_dataset, test_dataset = download_MNIST()

    # Convert Dataset to 0/1 binary
    convert_data_to_01(dataset=train_dataset, threshold=100, kind='train')
    convert_data_to_01(dataset=test_dataset,  threshold=100, kind='test')

    # Load 0/1 binary dataset
    X_train  = torch.load('./data/MNIST_binary/X_train.pt').to(torch.float32) 
    X_test   = torch.load('./data/MNIST_binary/X_test.pt').to(torch.float32)
    Y_train, Y_test = torch.load('./data/MNIST_binary/Y_train.pt'), torch.load('./data/MNIST_binary/Y_test.pt')

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE : {device}")
    
    #### DATASET ############################################################
    

    #### AUTOENCODER ########################################################
    
    # Load Autoencoder
    autoencoder = Autoencoder(input_dim=28*28, hidden_dim=args.auto_hidden_dim, encoded_dim=args.encoded_dim)
    autoencoder.to(device)

    print("\nTRAINING AUTOENCODER")
    # Train Autoencoder
    train_autoencoder(model=autoencoder, X_train=X_train, X_test=X_test, 
                    lr=args.auto_lr, num_epochs=args.auto_num_epochs, batch_size=args.auto_batch_size, device=device)
    
    # Load trained model
    print(f"The model is saved at : {autoencoder.ckpt_path}")
    autoencoder.load_pretrained_model(autoencoder.ckpt_path)
    
    print("IMAGE COMPARISION AFTER TRAINING AUTOENCODER\n")
    for idx in range(3):
        plot_original_vs_encoded(x=X_test[idx], model=autoencoder)
        
    #### AUTOENCODER ########################################################
    
    
    #### ENERGY BASED MODEL #################################################
    
    energy_model = EnergyNet(in_dim=autoencoder.encoded_dim, autoencoder_ckpt_path=autoencoder.ckpt_path ,hid_dim=args.ebm_hidden_dim)
    energy_model.to(device)
    
    # Note that for training only 4096 of the total points are used as it takes a lot of time to train
    # However in the notebook inference.ipynb, the best model is loaded that is trained on all data.
    train_ebm(energy_model=energy_model, autoencoder=autoencoder, X_train=X_train[:4096], X_test=X_test, 
              device=device, batch_size=args.ebm_batch_size, num_epochs=args.ebm_num_epochs, lr=args.ebm_lr, 
              mcmc_samples_per_datapoint=args.mcmc_samples_per_datapoint)
    
    # Load the trained model
    print("THE ENERGY MODEL IS TRAINED SUCESSFULLY")
    print(f"The Energy model is saved at : {energy_model.ebm_ckpt_path}")
    
    #### ENERGY BASED MODEL #################################################

    

if __name__ == "__main__":
    main()

    



