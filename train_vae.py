import torch
from model.vae import VAE_denoiser
from model.loader_utils import *


batch_size=256
learning_rate=1e-3

# Define the VAE model
VAE = VAE_denoiser()
VAE


# initialize loss function and optimizer
adam = torch.optim.Adam(VAE.parameters(), lr=learning_rate)

# Create DataLoader object to generate minibatches
train_dataset = BigwWigdDataset(input_dir='/hpcnfs/scratch/DP/amariani/Amariani/denoising_ChIPseq/data/tf', 
								bg_file='chrom1.bedGraph')

train_DT = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

trained_model = train_model(train_DT=train_DT, 
							model=VAE,
							optimizer=adam,
							num_epochs=15)

torch.save(trained_model, '/hpcnfs/scratch/DP/amariani/Amariani/denoising_ChIPseq/VAE_model.pth')