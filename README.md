# MegaDL: A python based machine learning software for metagenomic analysis to identify and predict disease sample accurately using deep neural networks.
  
Machine learning has been utilized in many applications from biomedical imaging to business analytics. Machine learning is stipulated to be a strong method for diagnostics and even for determining therapeutics in future as we move to precision medicine. MegaDL provides an unprecedented opportunity to develop neural networks from metagenomic data available publicly as well as to perform classification of data samples based on the optimal model we developed. 

The description below walks you through the analysis of the ___ project (https://pubs.broadinstitute.org/diabimmune)

The general workflow is described in below.



### Pre-requisites:

* Python version 3.7.x
    * Download Python (>3.6.0) version from Python.org.
    * Windows: https://www.python.org/downloads/windows/
    * Mac OS X: https://www.python.org/downloads/mac-osx/
    * Linux: https://www.python.org/downloads/source/
	
* Metagenomics data in BIOM or OTU format with accompanying metadata file.
	* Use Kraken 2 http://ccb.jhu.edu/software/kraken2/ for taxonomic classification of sequences.

## Installing MegaDL:


## Getting Started



### Data Input ###

MegaDL can take both OTU table and BIOM file from popular metagenomic profiling tools, [Kraken2](http://ccb.jhu.edu/software/kraken2/) and [qiime](https://qiime2.org/).
MegaDL provides a set of pre-processed datasets for use in training. 

With Kraken2 installed, follow the instructions at https://github.com/DerrickWood/kraken2/wiki/Manual to generate a taxonomic profile of your 16S or WGS data.


### Training the Model ###

To train the model, first download this github repository to your local device using git clone or similar command.

Next, navigate to the scripts folder in a command line environment, then run the following python script:

```python DNN.py ../Data/Cirrhosis.csv ../Data/CirrhosisMetaData.csv gridsearch=False threshold=0 normalize=False feature_level=All learning_rate=0.00001 epochs=10 batch=50 dropout_rate=0 early_stop=10```

This will generate a model using default parameters and using the Cirrhosis training data. The model will be saved as Cirrhosis.pt for use with prediction.

There are several optional parameters that can be used to fine tune the model, they are described below:

The GridSearch parameters can be used to leverage randomized grid search for hyperparameter optimization of the model.

The threshold parameter is used to prune the data of abundances that fall below the threshold, which tends to increase the accuracy of the model.

The normalization parameter executes data normalization when set to true.

The feature level parameter determines which taxonomic levels to use for classification. Options are: 'All', 'Species', and 'Genus'

The learning rate parameter sets the learning rate of the neural network, the optimal learning rate will vary depending on the dataset used.

The epochs parameter determines for how many iterations of the data the neural network will be trained on. Longer epochs will increase run-time and increase risk of 
overfitting.

The batch parameter sets what batch size to use for training the network.

The dropout rate determines the frequency at which the model will reset weights to help prevent overfitting.

The early stop feature is used only if grid search is set to true, and adds a limit for how many iteration of grid search are done without improvement before training is stopped.



### Prediction with trained model ###

To predict an unknown profile using a trained model, run the following command.
```python predict.py Cirrhosis.pt Cirrhosistest.csv```

This will return a prediction based on the trained model used. MegaDL provides a set of pretrained models for quick analysis of several datasets. 

#### Criteria for feature selection ####
**Genus Level** and **Species Level** tabs return genus and species level from the dataset as the feature. **All Level** tab tracks back the taxon level for unclassified higher order.  


#### Threshold ####
This field is getting a floating number to remove profiles and their abundances below the threshold value. Default value is **0**. 



#### Normalization ####
There is a choice for normalizing the data. Normalization is achieved using the cumulative sum scaling (CSS) method. 
