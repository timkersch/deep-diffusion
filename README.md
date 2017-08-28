# deep-diffusion

## Structure
- data
	* gen -  holds data generated from the camino toolkit. 
	Each subfolder has a config file with info about with what settings it was generated.
	* hpc_scanned_voxels.Bfloat - A big file with HPC scanned voxels (the ground truth) 
- models - Models are saved here
	* model1 - The model that performed best on the test-set. 
	Holds a config file with its setting, a plot of loss vs epoch, a pickled model, output from the run, and two scatter plots of prediction vs target for training and validation set.
- networks
	* fc_network.py - The network used for fitting the model
- config.json - This file is used to configure the network and the optimizer. See below.
- dataset.py - Loads and splits the datset into a validation and a training set
- generate_data.py - Has methods that calls a bash script that then calls the camino toolkit for generating data. 
- hpc.scheme - The schemefile used by the camino toolkit to generate data
- main.py - Parses the command line and calls appropriate methods
- run_camino.sh - The shell script calling camino toolkit
- run_param_eval.py - Methods for running knn on the training set and predicting on HPC data
- utils.py - Several helper functions for plotting and loading / parsing and plotting data


## Configuration (config.json)

- batch_norm - use batch-normalization in all layers except last
- batch_size - Size of each minibatch
- scale_inputs - Scale inputs to be between -1 and 1
- scale_outputs - Scale outputs to be between -1 and 1
- normalize - Set normalization with mean and/or standard deviation
- loss - Loss function to use during optimization, l1 or l2
- activation_function - Activation function to use, relu, sigmoid or tanh
- hidden_layers - List of layers including type "fc" for fully connected or "dropout" for dropout layer. Also need number of "units" if FC layer
- no_dwis - Number of diffusion weighted images per voxel in the input data. HPC is 288
- no_epoch - For how many epochs to run
- early_stopping - 0 to disable early stopping. n > 0 checks validation loss every n:th epoch to see if it has increased and stops if it has
- optimizer - type: adam or momentum. Parameters follows lasagna naming


## Running

### Data generation
Generate data with 100 iterations and 1000 voxels in every iteration

**-i** no. iteraions
**-v** no. voxels

`python main generate -i 100 -v 1000`

### Training
Train with ./config.json as config file and save model in ./models/model2

**-m** where to save model
**-c** path to config file to use

`python main training -m ./models/model2 -c ./config.json`


### Inference
Perform inference on voxels.bfloat with model.p, save output to predictions.txt

**-d** path to data in float32 bfloat format
**-m** path to model to perform inference
**-f** file on which to save the outputs

`python main inference -d ./data/voxels.bfloat -m ./models/model1/model.p -f ./predictions.txt`

### Search
Search for hyperparameters defined in parameter_search() in main.py

`python main search`







