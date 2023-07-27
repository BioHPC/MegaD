# MegaD: A package for metagenomic analysis to identify and predict disease sample accurately using deep neural networks.
  
Machine learning has been utilized in many applications from biomedical imaging to business analytics. Machine learning is stipulated to be a strong method for diagnostics and even for determining therapeutics in future as we move to precision medicine. MegaD provides an unprecedented opportunity to develop neural networks from metagenomic data available publicly as well as to perform classification of data samples based on the optimal model we developed. 

For a more in-depth understanding of the applications and methodologies of MegaD, please refer to our published article: (https://doi.org/10.3390/life12050669)

The description below walks you through the analysis of the DIABIMMUNE project (https://pubs.broadinstitute.org/diabimmune)

The general workflow is described in below.



### Pre-requisites:

* Python version 3.7.x
    * Download Python (>3.6.0) version from Python.org.
    * Windows: https://www.python.org/downloads/windows/
    * Mac OS X: https://www.python.org/downloads/mac-osx/
    * Linux: https://www.python.org/downloads/source/
	
* Metagenomics data in BIOM or OTU format with accompanying metadata file.
	* Use Kraken 2 http://ccb.jhu.edu/software/kraken2/ for taxonomic classification of sequences.

## Installing MegaD:
Before using our tool, several Python packages are required. These can be installed using the following commands. In the process of installing PyTorch, please refer to the provided link and proceed with the installation that best suits your system configuration. Choose the installation option that corresponds to your hardware capabilities, specifically whether you are using CUDA or not, depending on the presence or absence of a GPU in your system:
 - pytorch website: https://pytorch.org/get-started/locally/
 - CUDA website: https://developer.nvidia.com/cuda-toolkit-archive
```
>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
>pip install scikit-learn
>pip install tensorflow
>pip install matplotlib
>pip install numpy
>pip install pandas
```
Once these packages are installed, continue onto the next step to install our tool.

Installing MegaD is simple. There are two options for installing our software tool. 

#### [1] Gitclone installation:
For this installation, simply clone this repository into your location of choice using the following command:
```
>git clone https://github.com/BioHPC/MegaD.git
```
![2gif](https://github.com/BioHPC/MegaD/blob/master/pictures/2gif.gif)
#### [2] Manual installation:
Download the .zip file from the github page in the top right, then extract the contents of the file into your directory of choice.

## Getting Started



### Data Input ###

MegaD can take both OTU table and BIOM file from popular metagenomic profiling tools, [Kraken2](http://ccb.jhu.edu/software/kraken2/) and [qiime](https://qiime2.org/).
MegaD provides a set of pre-processed datasets for use in training. 

With Kraken2 installed, follow the instructions at https://github.com/DerrickWood/kraken2/wiki/Manual to generate a taxonomic profile of your 16S or WGS data.


### Training the Model ###

To train the model, navigate to the scripts folder in a command line environment, then run the following python script:

```
python DNN.py ../Data/dataset.csv ../Data/metadata.csv --threshold=0.03 -lr=0.001 --normalize=True --feature_level=Species --epochs=20
```
![1gif](https://github.com/BioHPC/MegaD/blob/master/pictures/1gif.gif)

This will generate a model using default parameters and using the selected training dataset. The model will be saved as dataset.pt for use with prediction.

There are several optional parameters that can be used to fine tune the model, they are described below:

* The GridSearch parameters can be used to leverage randomized grid search for hyperparameter optimization of the model.

* The threshold parameter is used to prune the data of abundances that fall below the threshold, which tends to increase the accuracy of the model.

* The normalization parameter executes data normalization when set to true.

* The feature level parameter determines which taxonomic levels to use for classification. Options are: 'All', 'Species', and 'Genus'

* The learning rate parameter sets the learning rate of the neural network, the optimal learning rate will vary depending on the dataset used.

* The epochs parameter determines for how many iterations of the data the neural network will be trained on. Longer epochs will increase run-time and increase risk of 
overfitting.

* The batch parameter sets what batch size to use for training the network.

* The dropout rate determines the frequency at which the model will reset weights to help prevent overfitting.

* The early stop feature is used only if grid search is set to true, and adds a limit for how many iteration of grid search are done without improvement before training is stopped.



### Prediction with trained model ###

To predict an unknown profile using a trained model, run the following command.
```python predict.py dataset.pt testdata.csv```

This will return a prediction based on the trained model used. MegaD provides a set of pretrained models for quick analysis of several datasets. 

#### Criteria for feature selection ####
**Genus Level** and **Species Level** tabs return genus and species level from the dataset as the feature. **All Level** tab tracks back the taxon level for unclassified higher order.  


#### Threshold ####
This field is getting a floating number to remove profiles and their abundances below the threshold value. Default value is **0**. Increasing this threshold can increase prediction accuracy by filtering out irrelevant taxonomies.



#### Normalization ####
There is a choice for normalizing the data. Normalization is achieved using the cumulative sum scaling (CSS) method. 

#### Gridsearch ####
This option repeatedly trains models using different parameters each time in order to determine the parameters that provide the highest prediction accuracy for each dataset. The grid search will stop once the number of runs without an accuracy improvement matches the value of the **Early stop** parameter.

#### Early Stop ####
Early stop defines how many epochs without improvement will occur before the gridsearch is stopped and a trained model is returned.

#### Dropout Rate ####
Drop out is the random resetting of weights during training, which can aid in reducing overfitting of the neural network.

#### Learning Rate ####
This parameter sets the learning rate to be used during the training process.

#### Epochs ####
This parameter defines how many training cycles are completed dueing the training process.

## End to end example

For this example, we will demonstrate and example usage of our model using the purina dataset to obtain the best results possible. 

### Training ###

First, we will train the model on the Purina.csv dataset using the following command:
```
>python DNN.py ../Data/wgs_purina.csv ../Data/wgs_purina_metadata.csv --threshold=0.03 --normalize=True --feature_level=Species --epochs=20
```
Alternatively, if we would like the best results possible we can utilize the grid search function, this will greatly increase the time it takes to train the model, but will produce better results. We can do that using the following command:
```
>python DNN.py ../Data/wgs_purina.csv ../Data/wgs_purina_metadata.csv --threshold=0.03 --normalize=True --feature_level=Species --epochs=20 --gridsearch=True
```

This will generate the following:

* An ROC curve that can be saved for future use
* A prinout of training accuracy, validation accuracy, testing accuracy, and testing AUC in the console

### Prediction ###

In order to sample MegaD's ability to predict disease status from taxonomic profile, we will run the prediction using the same Purina.csv dataset.

To predict, run the following command:
```
>python DNN.py ../Purina.csv ../wgs_purina_metadata.csv ../Purina.csv --threshold=0.03 --normalize=True --feature_level=Species --epochs=10 > prediction.txt
```

This will output the model's predictions to a file named prediction.txt for further analysis.
