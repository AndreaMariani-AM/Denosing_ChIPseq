import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np


# torch.utils.data.Dataset stores samples and the corresponding label
# torch.utils.data.DataLoader wraps an interable around Dataset to enable easy access to the samples

# class Dataset must implement three methods: __init__, __len__, __getitem__

# make a custom Dataset class
# define the class arrays_from_file as a dataloader
class BigwWigdDataset(Dataset):
	def __init__(self, input_dir, bg_file, Train= True, transform=None, target_transform=None):
		self.train = Train
		self.input_dir = input_dir
		self.bg = pd.read_csv(os.path.join(self.input_dir, bg_file), 
						sep='\t', names=['chrom', 'start', 'end', 'score'])
		self.df = self._get_data(Train)
		self.transform = transform
		self.target_transform = target_transform
	
	def _get_data(self, train):
		# set the dictionary of chromosomes
		get_chroms = self._get_chroms()
		# create masks
		mask_test = self.bg['chrom'].isin(get_chroms[0]) # 0 is the index of test chroms
		mask_validation = self.bg['chrom'].isin(get_chroms[1]) # 1 is the index of validation chroms
		mask_remove = self.bg['chrom'].isin(get_chroms[2]) # 2 is the index of to_remove chroms
		# cleanup the bedGraph
		cleaned_bg = self.bg[~mask_remove]
		

		if train == True:
			#get training samples
			data = cleaned_bg[~mask_test]
		else:
			data = cleaned_bg[mask_test]
		
		return data
		
	
	def __len__(self):
		return(len(self.df))
	
	def __getitem__(self, idx):
		#retrieves values
		# shape = (round(len(self.bg['score'].values) / (self.dims[0]*self.dims[1])), self.dims[0], self.dims[1])
		# end_index = shape[0]*shape[1]*shape[2]
		# reshaped = np.reshape(np.array(self.bg['score'].values[:end_index], dtype=np.float32), shape)
		# chip_values = reshaped[idx]

		chip_values = np.array(self.df.iloc[idx, 3], dtype=np.float32) #coerce to float32 for the linear layer dtype
		#norm_values = np.array(np.sinh(chip_values))
		
		#make it into a tensor
		chip = torch.from_numpy(chip_values)
		# create the label
		#start = self.bw.intervals(self.chrom)[idx][0]
		#end = self.bw.intervals(self.chrom)[idx][1]
		#label = str(self.chrom) + "_" + str(start) + "_" + str(end)
		#transformations
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		
		return chip

	def _get_chroms(self):
		# retrieve chroms for train/validation and test
		# hardcoded chromosomes for now
		test = [
			'chr1', 
		  	'chr2'
		  	]
		validation = [
			'chr3', 
			'chr4'
			]
		to_remove = [
			'chrY', 
			'chrM'
			]

		lists = [test, validation, to_remove]
		return lists



# Define the loss function
def loss_function(x_hat, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.5 * kl_loss
	# another idea
	# return recon_loss + kl_loss


def train_model( 
	train_DT,
	model,
	optimizer,
    num_epochs=15,
  ):
  size = len(train_DT.dataset)
  model.train()
  # Train the model
  for epoch in range(num_epochs):
      epoch_loss = 0
      for batch, X in enumerate(train_DT):
		  
        # Zero the gradients
        optimizer.zero_grad()

        # Get batch
        x = X

        # Forward pass
        x_hat, mu, logvar = model(x)
        # print(f"this is x_hat : {x_hat}\n-------------")
        # print(f"this is x : {x}\n-------------")
        # print(f"this is mu : {mu}\n-------------")
        # print(f"this is logvar : {logvar}\n-------------")

        # Calculate loss
        loss = loss_function(x_hat, x, mu, logvar)


        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Add batch loss to epoch loss
        epoch_loss += loss.item()

      # Print epoch loss
      print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/len(X)}")
      
  return model



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# # Alfredo's implementation

# # Define the loss function
# def loss_function(x_hat, x, mu, logvar):
#     recon_loss = nn.functional.mse_loss(x_hat, x.view(-1, 1000), reduction='sum')
#     kl_loss = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
#     return recon_loss + kl_loss
# 	# another idea
# 	# return recon_loss + kl_loss


# def train_model( 
# 	train_DT,
# 	model,
# 	optimizer,
#     num_epochs=15,
#   ):
#   # Train the model
#   for epoch in range(0, num_epochs + 1):
#       if epoch > 0: #test untrained net first
#         model.train()
#         epoch_loss = 0
		
#         for _, X in enumerate(train_DT):
#             # Get Tensor
#             # X = X.view(X.size(0), -1)
#             # X /= X.max(1, keepdim=True)[0]
#             # x = X.view(10, 10, 100) 
#             x = X
#             #===========FORWARD===================
#             x_hat, mu, logvar = model(x)
#             # Calculate loss
#             loss = loss_function(x_hat, x, mu, logvar)
#             # Add batch loss to epoch loss
#             epoch_loss += loss.item()

#             # print(f"this is x_hat : {x_hat}\n-------------")
#             # print(f"this is x : {x}\n-------------")
#             # print(f"this is mu : {mu}\n-------------")
#             # print(f"this is logvar : {logvar}\n-------------")

#             #===========BACKWARDS===================
#             # Zero the gradients
#             optimizer.zero_grad()
#             # Backward pass
#             loss.backward()
#             # Update parameters
#             optimizer.step()

# 	    #===========LOGS===================	
#         # Print epoch loss
#         print(f"=====> Epoch {epoch}, Average Loss: {epoch_loss/len(train_DT.dataset):.4f}")
      
#   return model