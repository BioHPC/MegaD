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

To train the model, run the following command:
```model = RunAnalysis(GridSearch = True, Threshold = 0.05, Normalize = True, level = 'All')```
The GridSearch parameters can be used to levrage randomized grid search for hyperparameter optimization of the model.
The threshold parameter is used to prune the data of abundances that fall below the threshold, which tends to increase the accuracy of the model.
The normalization parameter executes data normalization when set to true.
The level parameter determines which taxonomic levels to use for classification. Options are: 'All', 'Species', and 'Genus'

After entering the command, you will be prompted to enter the file names of your training data and metadata file.

### Prediction with trained model ###

To predict an unknown profile using a trained model, run the following command.
```model.predict('testing_data.csv')```

This will return a prediction based on the trained model used. MegaDL provides a set of pretrained models for quick analysis. 

#### Criteria for feature selection ####
**Genus Level** and **Species Level** tabs return genus and species level from the dataset as the feature. **All Level** tab tracks back the taxon level for unclassified higher order.  


#### Threshold ####
This field is getting a floating number to remove profiles and their abundances below the threshold value. Default value is **0**. 



#### Normalization ####
There is a choice for normalizing the data. Normalization is achieved using the cumulative sum scaling method. 