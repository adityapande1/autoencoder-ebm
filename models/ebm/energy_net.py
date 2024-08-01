import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ebm.energy_net_utils import langevin_MCMC


class EnergyNet(nn.Module):
    """
    Simple Energy based Neural Net, Outputs are assumed positive
    """
    def __init__(self, in_dim, hid_dim, autoencoder_ckpt_path, out_dim=1, dropout_prob=.1):
        super(EnergyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.in_dim = in_dim
        self.autoencoder_ckpt_path = autoencoder_ckpt_path      # Each ebm has its specific autoencoder

    def forward(self, x):
        
        # Forward Pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)        
        x = F.relu(self.fc3(x))
        
        return x
    
    def load_pretrained_model(self, model_ckpt_path):
        """
        Sets the model dictionary to model_ckpt_path
        
        Args:
            model_ckpt_path (str): checkpoint path (.pth file)
        """
    
        self.load_state_dict(torch.load(model_ckpt_path))
        print("Model Loaded Sucessully")
        
        self.ebm_ckpt_path = model_ckpt_path
    
    
    def generate_images(self, autoencoder, device, num_images=10):
    
        x_sample = langevin_MCMC(f_theta=self, input_dim=autoencoder.encoded_dim,  
                                batch_size=num_images, num_steps=10000, eps=.001)

        generated_data = autoencoder.decoder.predict(x_sample.to(device))
        generated_data = generated_data.detach().to('cpu')
        generated_data = generated_data.reshape(-1,28,28)
            
        return generated_data




