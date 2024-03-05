# This modified script handles bigwig files created with deeptools
# where contiguous bins with the same scores are merged together

# These functions have been adapted from Jacob Schreiber's Avocado (https://github.com/jmschrei/avocado/blob/master/avocado/utils.py)
# All credits for these functions go to him


import numpy as np
import pandas as pd
import os
from tqdm import tqdm

chroms = range(1, 20)

def create_bedgraph(name, chip_dir ,chroms=chroms, chrom_lengths=None, verbose=True):
	chrom_data = []

	for chrom in chroms:
		name = name.split('/')[-1]
		bedgraph = name + '.chr{}.bedgraph'.format(chrom)
		if verbose == True:
			print("bigWigToBedGraph {} {} -chrom=chr{}".format(name, 
				bedgraph, chrom))
		
		os.system("bigWigToBedGraph {}/{} {}/{} -chrom=chr{}".format(chip_dir, name, 
			chip_dir,bedgraph, chrom))

		if verbose == True:
			print("bedgraph_to_dense({})".format(bedgraph))
		
		data = bedgraph_to_dense(bedgraph, chip_dir,verbose=verbose)

		if verbose == True:
			print("decimate_vector")

		data = decimate_vector(data)

		if chrom_lengths is not None:
			if chrom != 'X':
				data_ = numpy.zeros(chromosome_lengths[chrom-1])
			else:
				data_ = numpy.zeros(chromosome_lengths[-1])

			data_[:len(data)] = data
			data = data_

		chrom_data.append(data)

		if verbose == True:
			print("rm {}".format(bedgraph))
		
		os.system("rm {}/{}".format(chip_dir, bedgraph))

	# if verbose == True:
	# 	print("rm {}".format(name))

	# os.system("rm {}".format(name))
	return chrom_data

def bedgraph_to_dense(filename, dir, verbose=True):
	"""
	Read a bedgraph file and return a dense numpy array.
	"""
	tmp_path = dir + '/' + filename
	bedgraph = pd.read_csv(tmp_path, sep="\t", header=None)
	n = bedgraph[2].values[-1]
	k = bedgraph.shape[0]
	data = np.zeros(n)
	
	# save to bedgraph
	#scores = decimate_vector(data)
	# create columns for df
	#df = create_columns()

	d = not verbose
	for i, (_, start, end, v) in tqdm(bedgraph.iterrows(), total=k, disable=d):
		data[start:end] = v

	return data

def decimate_vector(x, k=25):
	"""
	Reduce size of the vector to a 25bp window average value
	"""

	m = x.shape[0] // k
	y = np.zeros(m)

	for i in range(m):
		y[i] = np.mean(x[i*k:(i+1)*k])
	
	return y