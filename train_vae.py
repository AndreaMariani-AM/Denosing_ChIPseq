import torch
from model.vae import VAE_denoiser
from model.loader_utils import *
from model.bigwig_handler import *

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set params
batch_size=128
learning_rate=1e-4
weight_decay=1e-7

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the VAE model
VAE = VAE_denoiser().to(device)
VAE


# initialize loss function and optimizer
adam = torch.optim.Adam(VAE.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create DataLoader object to generate minibatches
train_dataset = BigwWigdDataset(input_dir='/hpcnfs/scratch/DP/amariani/Amariani/denoising_ChIPseq/data/tf', 
								bg_file='ring_train.npy')
test_dataset = BigwWigdDataset(input_dir='/hpcnfs/scratch/DP/amariani/Amariani/denoising_ChIPseq/data/tf', 
								bg_file='jarid2_test.npy')

train_DT = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_DT = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


losses_train, losses_test, kl_train, trained_model = train_test_model(train_DT=train_DT,
																		test_DT=test_DT,
																		model=VAE,
																		optimizer=adam,
																		device=device,
																		num_epochs=20)


#torch.save(trained_model, '/hpcnfs/scratch/DP/amariani/Amariani/denoising_ChIPseq/VAE_model.pth')
