# FourierHashNet

[Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search (NeurIPS 23, Spotlight)](https://indradyumna.github.io/pdfs/2023_FourierHashNet.pdf)

This directory contains code necessary for running all the experiments.

## Requirements

Recent versions of Pytorch, numpy, scipy, sklearn and matplotlib are required.
Additional third party softwares used - Dr.Hash, SBERT  
You can install all the required packages using  the following command:

	$ pip install -r requirements.txt

#Datasets and trained models
Please download files from https://rebrand.ly/fhash and place in the current folder. 
This contains the original datasets, dataset splits, trained models and other intermediate data dumps for reproducing tables and plots.  


## Run Experiments

The command lines and scripts used for training models are in scripts/ folder.   
Command lines specify the exact hyperparameter settings used to train the models.   

## Reproduce plots and figures  

FinalSubmission-Figs-NeurIPS23-Main.ipynb contains code used for generating figures from the paper .   

Notes:  
 - GPU usage is required for model training
 - Hashing is done only on CPU. 
 - source code files are all in src folder.  


Reference
---------

If you find the code useful, please cite our paper:

	@article{roy2023locality,
	  title={Locality sensitive hashing in fourier frequency domain for soft set containment search},
	  author={Roy, Indradyumna and Agarwal, Rishi and Chakrabarti, Soumen and Dasgupta, Anirban and De, Abir},
	  journal={Advances in Neural Information Processing Systems},
	  volume={36},
	  pages={56352--56383},
	  year={2023}
	}

Indradyumna Roy, Indian Institute of Technology - Bombay  
indraroy15@cse.iitb.ac.in
