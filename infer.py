import argparse

import matplotlib.pyplot as plt
import torch

from models.autoencoder.autoencoder import Autoencoder
from models.ebm.energy_net import EnergyNet
from utils import plot_digits, plot_original_vs_encoded


def main():
    
    parser = argparse.ArgumentParser(description="Parse model training parameters")    
    parser.add_argument('--num_images', type=int, required=True, help='Number of images to be generated')
    args = parser.parse_args()

    #### AUTOENCODER ########################################################
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE : {device}")
    
    # Load Autoencoder
    autoencoder = Autoencoder(input_dim=28*28, hidden_dim=1024, encoded_dim=4)
    autoencoder.to(device)

    # Load pretrained best checkpoint
    best_autoencoder_ckpt_path = './saved_ckpts/autoencoder/autoencoder_best.pth'
    autoencoder.load_pretrained_model(model_ckpt_path=best_autoencoder_ckpt_path)
        
    #### AUTOENCODER ########################################################
    
    
    #### ENERGY BASED MODEL #################################################
    
    # Load EBM
    energy_model = EnergyNet(in_dim=autoencoder.encoded_dim, autoencoder_ckpt_path=autoencoder.ckpt_path, hid_dim=8)
    energy_model.to(device)

    # Load pretrained best checkpoint
    best_ebm_ckpt_path = './saved_ckpts/ebm/ebm_best.pth'
    energy_model.load_pretrained_model(model_ckpt_path=best_ebm_ckpt_path)
    
    # Load the trained model
    print("THE ENERGY MODEL IS LOADED SUCESSFULLY")
    print(f"The Energy model is loaded from : {energy_model.ebm_ckpt_path}")
    
    #### ENERGY BASED MODEL #################################################
    
    ############ GENERATION #################################################
    
    # Generate images from energy model
    generated_images  = energy_model.generate_images(autoencoder=autoencoder, num_images=args.num_images, device=device)

    # Visualise the generated images
    plot_digits(X=generated_images, Y=None, n=args.num_images)
    
    ############ GENERATION #################################################
    

if __name__ == "__main__":
    main()

    
    




