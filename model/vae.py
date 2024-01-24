import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

class VAE_denoiser(nn.Module):
    def __init__(
          self, 
          x_dim=128,
          hidden_dim=64,
          z_dim=10
        ):
        super(VAE_denoiser, self).__init__()

        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_layer2_logvar = nn.Linear(hidden_dim, z_dim)

        # Define autoencoding layers
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
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# Alfredo's Canziani implementation

# class VAE_denoiser(nn.Module):
#     def __init__(
#           self
#         ):
#         super(VAE_denoiser, self).__init__()
        
#         d = 8

#         # Define autoencoding layers
#         self.encoder = nn.Sequential(
#                nn.Linear(128, d ** 2),
#                nn.BatchNorm1d(d**2),
#                nn.ReLU(),
#                nn.Linear(d ** 2, d*2)
#         )
        
#         self.decoder = nn.Sequential(
#                nn.Linear(d, d ** 2),
#                nn.BatchNorm1d(d**2),
#                nn.ReLU(),
#                nn.Linear(d ** 2, 128),
#                nn.ReLU()
#         )

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             # this is during training, adds the noise during sampling
#             std = torch.exp(logvar/2)
#             eps = torch.randn_like(std)
#             return mu + std * eps
        
#     def forward(self, y):
#         d = 8
#         mu_logvar = self.encoder(y.view(-1, 128)).view(-1, 2, d)
#         mu = mu_logvar[:, 0, :]
#         logvar = mu_logvar[:, 1, :]
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decoder(z)
#         return x_hat, mu, logvar
