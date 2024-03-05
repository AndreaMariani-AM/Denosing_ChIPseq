import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


######### VAE ###########
class VAE_denoiser(nn.Module):
    def __init__(
          self, 
          x_dim=1000,
          hidden_dim=256,
          z_dim=10
        ):
        super(VAE_denoiser, self).__init__()

        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_layer2_logvar = nn.Linear(hidden_dim, z_dim)

        # Define decoding layers
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim) 

    def encoder(self, x):
        x = F.relu(self.enc_layer1(x))
        mu = F.relu(self.enc_layer2_mu(x))
        logvar = F.relu(self.enc_layer2_logvar(x))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            # this is during training, adds the noise during sampling
            std = torch.exp(logvar/2)
            eps = torch.randn_like(std)
            z = mu + std * eps
            return z
        else:
            # this is done during inference, i turn off the noise
            return mu
        
    def decoder(self, z):
        # Define decoder network
        output = F.relu(self.dec_layer1(z))
        output = self.dec_layer2(output)
        return output

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 1000))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    

####### Denoising AE #########
class denoising_AE(nn.Module):
    def __init__(
            self,
            x_dim=1000,
            hidden_dim=256,
            z_dim=10
            ):
        super(denoising_AE, self).__init_()

        # Encoder Layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2 = nn.Linear(hidden_dim, z_dim)
        
        # Decoder Lyares
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim)
        
    # Encoder 
    def encoder(self, x):
            x 

        # Deco
        # Forward

        