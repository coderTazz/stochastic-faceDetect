# Face Detection using Probability-Based Generative Models

Using the FDDB dataset to create positive (face) and negative(non-face) patches.

Will be using the following models:
*	Multivariate Gaussian
*	Mixture of Multivariate Gaussians
*	Multivariate t-distribution
*	Multivariate Factor Analyzers


## Running Instructions

It is assumed that you are inside the stochastic-FaceDetect folder.

### Preparing Data
*	Download the FDDB dataset(link in reference) and extract the two folders.
*	Make another folder 'savedPics' containing two sub-folders 'train' and 'test'. 
*	Both of these have two subfolders each: 'pos' and 'neg'
*	Place 'savedPics' outside
*	Resolution of the patches can be set from inside the file (default 20) 

`cd src`

`python3 prepareFDDBData.py`

Note that the 20x20x3 patches are already present in this repo. The above instructions need to be followed only if creating new patches. Else, proceed to the main experiment.

### Main Experiment
*	Options for model_type:
	1. 	`'gaus'`: Multivariate Gaussian
	2. 	`'gmm'`: Mixture of Multivariate Gaussians
	3. 	`'tdst'`: Multivariate t-distribution
	4. 	`'fcta'`: Multivariate Factor Analyzers

*	Training Size Max: 8000
*	Testing Size Max: 2000



`cd src`

`python3 main.py --tr_sz=<insert training data size, default 2000> --te_sz=<insert testing data size, default 200> --model_type=<insert model type, default 'gaus'>`



## References
*	[FDDB: A Benchmark for Face Detection in Unconstrained Settings](http://vis-www.cs.umass.edu/fddb/fddb.pdf)