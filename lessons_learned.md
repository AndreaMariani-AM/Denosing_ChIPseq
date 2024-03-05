# Lessons Learned

## Exploding Gradients

I was tryng to figure out why predictions (and loss) went to NaN right after the second batch. $\mu$ and $\log\sigma$ went straight to the moon. Apparently this is a classic example of exploding gradients. What i can try is:
1) Batch Normalization
2) Weights Initialization
3) Gradient Clipping

## Learning Rate

Nothing fancy here, just *TRIPLE* check that the learning rate is a actually the intended value!! I've just wasted three days because i thought i'd set it to $1e-3$ while instead it was $1e3$ and made the weights unstable.


## TO DO

1) Create temporary bedGraphs files for each ChIP. Decide if shoulb be split per chromosome or not based on size. chrom1 and 2 are for test and chrom 3 and 4 for validation
2) Assuming i want roughly 25kb of information and i have 50bp bins, that means ~ 500 "rows" of the bedgraph --> this equals one sample.    
	If i sample, let's say, 128 samples as a mini-batch i'd have ~ 17 batches per epoch per chromosome.