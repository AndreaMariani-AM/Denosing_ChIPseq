import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import math


# torch.utils.data.Dataset stores samples and the corresponding label
# torch.utils.data.DataLoader wraps an interable around Dataset to enable easy access to the samples

# class Dataset must implement three methods: __init__, __len__, __getitem__

# make a custom Dataset class
# define the class arrays_from_file as a dataloader
class BigwWigdDataset(Dataset):
	def __init__(self, input_dir, bg_file):
		self.input_dir = input_dir
		self.df_file = np.load(os.path.join(self.input_dir, bg_file))
		self.input_data = self._reshape_input()
	
	def __len__(self):
		return(len(self.input_data))

	def _reshape_input(self):
		x = len(self.df_file)
		a, remainder = divmod(x, 1000)  # Set chunks of 1000 elements
		if remainder > 0:
			a += 1

		# Fill up the array to the required length
		pad_length = a * 10 * 100 - x
		padded_array = np.pad(self.df_file, (0, pad_length), mode='constant')

		# Reshape the array to the desired shape
		result_array = padded_array.reshape((a, 10, 100))

		return result_array
	
	
	def __getitem__(self, idx):
		#retrieves values
		chip_values = np.array(self.input_data[idx], dtype=np.float32) #coerce to float32 for the linear layer dtype
		#make it into a tensor
		chip = torch.from_numpy(chip_values)
		# Normalize using Softmax (Min-Max or Mean 0 and std 1 are broken for a tensor full of zeros)
		softmax_fun = nn.Softmax(dim = 0)
		norm_values = softmax_fun(chip.view(-1)).view(10, 100) #softmax for the whole tensor and then put it back in the right shape
		
		return norm_values


# Define the loss function
def loss_function(x_hat, x, mu, logvar):
	recon_loss = nn.functional.mse_loss(x_hat, x.view(-1, 1000), reduction='sum')
	kl_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
	return kl_loss, recon_loss + kl_loss
	# another idea
    # return  kl_loss + math.sqrt(kl_loss*recon_loss) + recon_loss


def train_test_model( 
	train_DT,
	test_DT,
	device,
	model,
	optimizer,
	num_epochs=15
	):
	# Train and test the model
	losses_train = []
	kl_train = []
	losses_test = []
	for epoch in range(0, num_epochs+1):
		# Training
		if num_epochs > 0: # test untrained model first
			model.train()
			train_loss = 0
			kl = 0
			for _, X in enumerate(train_DT):
			
				# Get batch
				x = X.to(device)

				# =======Forward pass======= #
				x_hat, mu, logvar = model(x)
				# print(f"this is x_hat : {x_hat}\n-------------")
				# print(f"this is x : {x}\n-------------")
				# print(f"this is mu : {mu}\n-------------")
				# print(f"this is logvar : {logvar}\n-------------")

				# Calculate loss
				kl_loss, loss = loss_function(x_hat, x, mu, logvar)

				 # Add batch loss to epoch loss
				train_loss += loss.item()
				kl += kl_loss.item()

				# =======Backward======= #
				# Zero the gradients
				optimizer.zero_grad()
				# Backward pass
				loss.backward()
				# Update parameters
				optimizer.step()

			# =======Log======= #
			# Average loss
			avg_loss_epoch = train_loss / len(X)
			avg_kl = kl / len(X)
			# store the loss
			losses_train.append(avg_loss_epoch)
			kl_train.append(avg_kl)
			# Print epoch loss
			print(f"----> Epoch {epoch+1}/{num_epochs} | Loss_train: {train_loss/len(X):.4f}")
		
		# Testing
		#means, logvar = list(), list()
		with torch.no_grad():
			model.eval()
			test_loss = 0
			for _, X in enumerate(test_DT):
				# Get batch
				x = X.to(device)
				# =======Forward pass======= #
				x_hat, mu, logvar = model(x)
				# Loss
				_, tt_loss = loss_function(x_hat, x, mu, logvar)
				test_loss += tt_loss.item()
				# =======Log======= #
				#means.append(mu.detach())
				#logvar.append(logvar.detach())
		
		# average test loss
		avg_loss_test = test_loss / len(X)
		# store losses
		losses_test.append(avg_loss_test)
		# Print epoch test loss
		print(f"--------> Loss_test: {test_loss:.4f}")

			

	return losses_train, losses_test, kl_train, model




