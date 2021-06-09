import pandas as pd
import numpy as np
def process_data(dataset_path,metadata_path,level):
        data = pd.read_csv(dataset_path, sep=',', header=0) #Use input name +.csv to import data set to pandas dataframe. Pandas used for robustness of import function
        my_data = data.to_numpy() #Convert to numpy array
        transposed = my_data.transpose()#Transpose Numpy array for easier manipulation and feature extraction
        remove_indeces = []
        if level == 'Species':
                for x in range(len(transposed[0])):
                        if (not 's_' in str(transposed[0][x])) or ('t_' in str(transposed[0][x])):
                                remove_indeces.append(int(x))
                transposed = np.delete(transposed, remove_indeces, 1)

        elif level == 'Genus':
                for x in range(len(transposed[0])):
                        if (not 'g_' in str(transposed[0][x])) or ('s_' in str(transposed[0][x])):
                                remove_indeces.append(int(x))
                transposed = np.delete(transposed, remove_indeces, 1)
                               
        x_train=np.array(transposed[1:])        #Select columns as features
        x_train = x_train.astype('Float64')
        labels = pd.read_csv(metadata_path, sep=',', header=0)        #Import metadata file
        labels = labels.to_numpy()      #Convert pandas df to numpy for easier handling
        labels=labels.transpose()
        return x_train,labels
