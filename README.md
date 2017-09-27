# Simulating the ground truth in Diffusion MRI

Author: Tim Kerschbaumer (2017-08-31)

## General
Project in "Practical Course: Hands-on Deep Learning for Computer Vision and Biomedicine" at TU Munich SS 2017.

The purpose of the project was to simulate diffusion MRI data using the Camino Diffusion MRI Toolkit and then train a
neural network on the simulated data. The networks aim is to predict the configurations used to generate the data, i.e the inverse mapping.

Please see './presentation/Presentation.pdf' for details. 


## File structure
- data
	* gen - holds data generated from Camino Toolkit. 
	Each subfolder is one simulation run. Each run also has a config file with info about what settings were used. 
	Data in this directory is used to train the network.
	* search - holds data generated from Camino Toolkit that is used to find appropriate simulation ranges.
	Data in this directory is used to fit a kNN model.
	* hpc/50000_scanned_voxels.Bfloat - A file with 50000 scanned voxels from the HPC dataset
	* hpc/data.nii.gz - Not included in git repo, contains full diffusion MRI scan from HPC. Can be downloaded from HPC
	* hpc.scheme - Schemefile for Camino Toolkit simulation (from HPC dataset)
- models - Trained models are saved here
	* model1 - The model that performed best during hyperparameter search
		* config.json - the configuration for the model
		* loss-plot.png - plot of loss vs epochs
		* model.p - the saved model
		* out.txt - the output from network training
		* train-diff-plot.png - scatter plot of targets vs predictions for training set
		* train-residual-plot.png - scatter residual plot of training set
		* validation-diff-plot.png - scatter plot of targets vs predictions for validation set
		* validation-residual-plot.png - scatter residual plot of validation set
- networks
	* fc_network.py - The neural network class
- plots - Contains some plots  
	* hpc-heat-plots - Shows heat plots of predicted radiuses on each z-slice on a full HPC dataset scan (data.nii.gz)
	* heat-plot-depth-vs-width shows a heat plot of R2 score on network depth vs height
	* heat-plot-dropout-vs-width shows a heat plot of R2 score on dropout factor vs network width
	* hpc-voxel-predictions-histogram shows a histogram of predicted radiuses on HPC data with kNN fitted to the search data
- config.json - This file is used to configure the network and the optimizer. See below.
- dataset.py - Loads and splits the dataset.
- generate_data.py - Used for calling a bash script that then calls the Camino Toolkit for generating data. 
- main.py - Parses the command line and calls appropriate methods.
- run_camino.sh - The bash script calling Camino Toolkit.
- run_param_eval.py - Methods for running kNN on data and predicting on HPC data to find appropriate simulation ranges.
- utils.py - Several helper functions for plotting, metrics and loading / parsing data
- presentation - Holds .pdf and LaTeX source of presentation as well as images


## Configuration (config.json)

- batch_norm - boolean - use batch-normalization in all layers except last
- batch_size - int - Size of each minibatch
- scale_inputs - boolean - Scale inputs to be between -1 and 1
- scale_outputs - boolean - Scale outputs to be between 0 and 1
- normalize - boolean - Set normalization with mean and/or standard deviation
- loss - 'l1' or 'l2' -  Loss function to use during optimization
- activation_function - 'relu', 'tanh' or 'sigmoid' - Activation function to use
- hidden_layers - List of layers including type "fc" for fully connected or "dropout" for dropout layer. Also need number of "units" if FC layer or "p" for dropout layer
- no_dwis - int - Number of diffusion weighted images per voxel in the input data. HPC is 288
- no_epoch - int - For how many epochs to run
- early_stopping - int - 0 to disable early stopping. n > 0 checks validation loss every n:th epoch to see if it has increased and stops if it has
- optimizer - 'adam' or 'momentum' - Parameters follows lasagna naming


## Data
- Generated data for training the network in /data/gen/ - (samples x features) = (93900 x 288)
- Generated data for kNN settings comparison with HPC in /data/search/ - (samples x features) = (21000 x 288)
- Sample HPC Data for kNN settings comparison in /data/hpc/50000_scanned_voxels.Bfloat - (samples x features) = (50000 x 288)
- All HPC Data for kNN settings comparison in /data/hpc/data.nii.gz - (shape) = (145 x 174 x 145 x 288)

All .bfloat datafiles are saved and loaded as binary float32 1-D arrays.


## Running
Please set the Theano flags environment parameters before executing scripts

floatX should be 'float32' and device should be either 'gpu' or 'cpu'

THEANO_FLAGS='device=gpu,floatX=float32'

### Data generation
`python main.py generate -i 100 -v 1000`

Generate data with 100 iterations and 1000 voxels in every iteration (requires Camino Toolkit to be installed and in $PATH)

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