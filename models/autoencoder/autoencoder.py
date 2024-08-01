import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoded_dim)
        )
    
    def forward(self, x):
        
        x = x.view(x.shape[0],-1)   # Keep the batch dimension, flatten all other
        
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Using sigmoid for binary data output
        )
    
    def forward(self, x):
        
        x = x.view(x.shape[0],-1)   # Keep the batch dimension, flatten all other
        
        return self.decoder(x)
    
    def predict(self, x, threshold=.5):
        
        self.decoder.eval()             # Set decoder to evaluation mode
        out = self.forward(x)           # Do a forward pass
        out = (out>=threshold).int()    # Filter 0/1 given the threshold
        
        return out

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, encoded_dim)
        self.decoder = Decoder(encoded_dim, hidden_dim, input_dim)
        self.encoded_dim = encoded_dim
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def load_pretrained_model(self, model_ckpt_path):
        """
        Sets the model dictionary to model_ckpt_path
        
        Args:
            model_ckpt_path (str): checkpoint path (.pth file)
        """
    
        self.load_state_dict(torch.load(model_ckpt_path))
        print("Model Loaded Sucessully")
        
        self.ckpt_path = model_ckpt_path
    
    


