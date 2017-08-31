# Simulating the ground truth in Diffusion MRI

Author: Tim Kerschbaumer (2017-08-31)

## General
Please set the Theano flags environment parameters before executing scripts

floatX should be 'float32' and device should be either 'gpu' or 'cpu'

THEANO_FLAGS='device=gpu,floatX=float32'

## File structure
- data
	* gen -  holds data generated from the camino toolkit. 
	Each subfolder is one simulation run. Each run also has a config file with info about with what settings it was generated.
	* search - holds data generated from camino toolkit but that is used to find appropraiate simulation settings
	* hpc/50000_scanned_voxels.Bfloat - A file with 50000 scanned voxels from the HPC dataset
	* hpc.scheme - Schemefile for Camino toolkit simulation (from HPC dataset)
- models - Models are saved here
	* model1 - The model that performed best during hyperparameter search
		* config.json - the configuration for the model
		* loss-plot.png - plot of loss vs epoch
		* model.p - the saved model
		* out.txt - the output during the network training
		* train-diff-plot.png - scatter plot of trargets vs predictions for training set
		* train-residual-plot.png - scatter residual plot of training set
		* validation-diff-plot.png - scatter plot of trargets vs predictions for validation set
		* validation-residual-plot.png - scatter residual plot of validation set
- networks
	* fc_network.py - The neural network class
- config.json - This file is used to configure the network and the optimizer. See below.
- dataset.py - Loads and splits the datset into a validation and a training set
- generate_data.py - Has methods that calls a bash script that then calls the camino toolkit for generating data. 
- main.py - Parses the command line and calls appropriate methods
- run_camino.sh - The shell script calling camino toolkit
- run_param_eval.py - Methods for running kNN on data and predicting on HPC data
- utils.py - Several helper functions for plotting and loading / parsing data
- presentation - Holds .pdf and LaTeX source of presentation as well as images


## Configuration (config.json)

- batch_norm - use batch-normalization in all layers except last
- batch_size - Size of each minibatch
- scale_inputs - Scale inputs to be between -1 and 1
- scale_outputs - Scale outputs to be between 0 and 1
- normalize - Set normalization with mean and/or standard deviation
- loss - Loss function to use during optimization, l1 or l2
- activation_function - Activation function to use, relu, sigmoid or tanh
- hidden_layers - List of layers including type "fc" for fully connected or "dropout" for dropout layer. Also need number of "units" if FC layer or "p" for dropout layer
- no_dwis - Number of diffusion weighted images per voxel in the input data. HPC is 288
- no_epoch - For how many epochs to run
- early_stopping - 0 to disable early stopping. n > 0 checks validation loss every n:th epoch to see if it has increased and stops if it has
- optimizer - type: adam or momentum. Parameters follows lasagna naming


## Data
- Generated data for training the network in /data/gen/ - (samples x features) = (93900 x 288)
- Generated data for kNN settings comparison with HPC in /data/search/ - (samples x features) = (11000 x 288)
- HPC Data for kNN settings comparison in /data/hpc/ - (samples x features) = (50000 x 288)

All .bfloat datafiles are saved and loaded as binary float32 1-D arrays.


## Running

### Data generation
`python main.py generate -i 100 -v 1000`

Generate data with 100 iterations and 1000 voxels in every iteration (requires camino to be installed an in $PATH)

**-i** no. iterations

**-v** no. voxels

### Training
`python main.py training -m ./models/model2 -c ./config.json`

Train with ./config.json as config file and save model in ./models/model2

**-m** where to save model

**-c** path to config file to use


### Inference
`python main.py inference -d ./data/hpc/50000_scanned_voxels.Bfloat -m ./models/model1/model.p -f ./predictions.bfloat`

Perform inference on 50000_scanned_voxels.Bfloat with model.p, save output to predictions.bfloat

**-d** path to data in float32 bfloat format

**-m** path to model to perform inference

**-f** file on which to save the outputs (float32 .bfloat format)

### Search
`python main.py search`

Search for hyperparameters defined in parameter_search() in main.py







