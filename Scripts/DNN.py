import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import sklearn
import sys
import argparse
import neuralnet
from DataProcessing import process_data
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from datetime import datetime
from torch.autograd import Variable
from torchvision import models




def read_params(args):
        parser = argparse.ArgumentParser(description='MegaDL - Disease status prediction of metagenomics samples using Machine Learning.')
        arg = parser.add_argument
        arg('inp_f', metavar = 'INPUT_DATA', nargs = '?',default=sys.stdin,type=str,help="The input dataset in BIOM.csv format")
        arg('inp_m', metavar = 'INPUT_META', nargs = '?', default=sys.stdin, type=str, help = "The inpute metadata file corresponding to input data in csv format.")
        arg('inp_u', metavar = 'UNKNOWN_INPUT', nargs='?', default=None, type=str,help="To predict an unknown metagenomic profile, enter it's path here in BIOM.csv format")
        arg('-grid_s', '--gridsearch',default = False, type=bool, help="Use randomized gridsearch for hyperparameter optimization of machine learning model")
        arg('-thresh', '--threshold',default = 0, type=float, help="Enter minimum abundance level for use in training model.")
        arg('-norm', '--normalize', default = False, type = bool, help = "Execute cumulative sum scaling normalization on training data.")
        arg('-level', '--feature_level', default = 'All', type = str, help = "Define taxonomic level to be used in model training and prediction.")
        arg('-lr', '--learning_rate', default = 0.00001, type=float, help="Enter the learning rate to use during training.")
        arg('-epochs', '--epochs', default = 10, type=int, help="Number of epochs to train on.")
        arg('-batch_size', '--batch', default = 50, type=int, help="Enter the desired batch size to use during training.")
        arg('-dr', '--dropout_rate', default = 0, type=float, help="Enter the desired dropout rate to use. Leave at default to disable dropout.")
        arg('-es', '--early_stop', default = 10, type=int, help="Applies to randomized grid search only. Specify how many iterations of no improvement before stopping the randomized grid search.")
        return vars(parser.parse_args())




##Function to run desired dataset

def Train_model(dataset_path,metadata_path,GridSearch=False, Threshold = 0, Normalize = False, level='All',epochs = 10,batch_size=50,learning_rate=0.00001,d_rate=0,early_stop=10):
        
        ##Initialize classifier

        #if torch.cuda.is_available():
                #torch.cuda.manual_seed(1)
                #usecuda=True
                #device = torch.device("cuda:0")

        ##Data processing
        x_train,labels = process_data(dataset_path, metadata_path, level=level)
        row_size = x_train.shape[1]
        if Normalize:
                x_train=normalize(x_train,axis=0,norm='l2')
                print('Normalizing!')
        unique = np.unique(labels[1]) #Gather all unique classes in labels
        label_dict = {}
        for i in range(len(unique)):    #Convert text labels to integer values
                label_dict[i]=unique[i]
                col = labels[1]
                col = np.where(col == unique[i], i , col)
                labels[1]=col
        classes = len(unique) #Determine number of classes in dataset

        #############################################################################################
        ##########################Place holder for threshold implementation##########################
        #############################################################################################

        x_train[x_train<Threshold] = 0.0

        ##Split Data into training and validation

        x_train,x_val,y_train,y_val = train_test_split(x_train, labels[1], test_size = 0.2, train_size = None, random_state=42, shuffle = True, stratify = labels[1])     #Split Data into training and validation
        x_val,x_test,y_val,y_test = train_test_split(x_val, y_val, test_size = 0.5, train_size = None, random_state=42, shuffle = True, stratify = y_val)       #Split validation into validation and testing
        #Convert value types to integer for accuacy comparison

        y_val = y_val.astype('int')     
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')
        x_train,y_train,x_val,y_val,x_test,y_test = torch.from_numpy(x_train),torch.from_numpy(y_train),torch.from_numpy(x_val),torch.from_numpy(y_val),torch.from_numpy(x_test),torch.from_numpy(y_test)
        ##Fit an initial model to the data using default parameters

        criterion = nn.CrossEntropyLoss()
        #if usecuda:
                #model.to(device)
        gridsearch_iters=30
        val_accuracies = []
        train_accuracies = []
        writer = SummaryWriter() #Initialize writer for creating graphs
        if not GridSearch:
                model = neuralnet.Net(row_size, 75, classes,15)
                optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0)
                model.train()
                for epoch in range(epochs):
                        train_pred = []
                        val_pred=[]
                        test_pred=[]
                        rnd_idx=np.random.permutation(x_train.shape[0]) #Generate batch indeces
                        for rnd_indices in np.array_split(rnd_idx,x_train.shape[0]//batch_size):

                        
                                x,target=Variable(x_train[rnd_indices]),Variable(y_train[rnd_indices])
                                #if usecuda:
                                        #x,target = x.to(device), target.to(device)
                                optimizer.zero_grad()
                                output=model(x.float()) #Generate output from model
                                loss=criterion(output,target.long()) #Calculate loss
                                loss.backward()
                                optimizer.step()
                        #Next three loops calculate training loss, validation loss, and testing loss for each epoch.
                        for row in range(x_train.shape[0]):
                                x,target=Variable(x_train[row]),Variable(y_train[row])
                                output = model(x.unsqueeze(dim=0).float())
                                train_pred.append(output.data.max(1,keepdim=True)[1])
                        for row in range(x_val.shape[0]):
                                x,target=Variable(x_val[row]),Variable(y_val[row])
                                output = model(x.unsqueeze(dim=0).float())
                                val_pred.append(output.data.max(1,keepdim=True)[1])
                        for row in range(x_test.shape[0]):
                                x,target=Variable(x_test[row]),Variable(y_test[row])
                                output = model(x.unsqueeze(dim=0).float())
                                test_pred.append(output.data.max(1,keepdim=True)[1])
                        test_accuracy = sklearn.metrics.accuracy_score(y_test,test_pred) 
                        val_accuracy = sklearn.metrics.accuracy_score(y_val,val_pred)
                        train_accuracy = sklearn.metrics.accuracy_score(y_train,train_pred)
                        #Add accuracies to graph
                        writer.add_scalar('Accuracy/val',val_accuracy,epoch)
                        writer.add_scalar('Accuracy/train',train_accuracy,epoch)
                        writer.add_scalar('Accuracy/test',test_accuracy,epoch)
                        train_accuracies.append(train_accuracy)
                        print("Epoch number:",epoch+1," Training accuracy:",train_accuracy,"Validation accuracy:",val_accuracy, "Testing accuracy:",test_accuracy)
                writer.close()
                return model
        else:
                count = 0
                best_accuracy = 0
                best_model = None
                for iteration in range(gridsearch_iters):
                        #Generate random parameters from set ranges
                        dropout_rate=0.5
                        n_layers = np.random.randint(5,75)
                        n_neurons = np.random.randint(5,75)
                        use_dropout=np.random.choice([True,False])
                        if use_dropout:
                                dropout_rate=np.random.uniform(0.1,0.9)
                        learning_rate = np.random.uniform(0.000005,0.00005)
                        weight_decay = np.random.uniform(0,0.1)
                        model = neuralnet.Net(row_size, n_neurons, classes, n_layers)
                        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
                        model.train()
                        #Train model on randomized parameters
                        for epoch in range(epochs):
                                rnd_idx=np.random.permutation(x_train.shape[0])
                                train_pred = []
                                val_pred=[]
                                test_pred=[]
                                for rnd_indices in np.array_split(rnd_idx,x_train.shape[0]//batch_size):

                        
                                        x,target=Variable(x_train[rnd_indices]),Variable(y_train[rnd_indices])
                                        #if usecuda:
                                                #x,target = x.to(device), target.to(device)
                                        optimizer.zero_grad()
                                        output=model(x.float())
                                        loss=criterion(output,target.long())
                                        loss.backward()
                                        optimizer.step()
                                #Calculate predictions for accuracy measurement
                                for row in range(x_train.shape[0]):
                                        x,target=Variable(x_train[row]),Variable(y_train[row])
                                        output = model(x.unsqueeze(dim=0).float())
                                        train_pred.append(output.data.max(1,keepdim=True)[1])
                                for row in range(x_val.shape[0]):
                                        x,target=Variable(x_val[row]),Variable(y_val[row])
                                        output = model(x.unsqueeze(dim=0).float())
                                        val_pred.append(output.data.max(1,keepdim=True)[1])
                                for row in range(x_test.shape[0]):
                                        x,target=Variable(x_test[row]),Variable(y_test[row])
                                        output = model(x.unsqueeze(dim=0).float())
                                        test_pred.append(output.data.max(1,keepdim=True)[1])
                                test_accuracy = sklearn.metrics.accuracy_score(y_test,test_pred)
                                val_accuracy = sklearn.metrics.accuracy_score(y_val,val_pred)
                                train_accuracy = sklearn.metrics.accuracy_score(y_train,train_pred)
                                writer.add_scalar('Accuracy/val',val_accuracy,epoch)
                                writer.add_scalar('Accuracy/train',train_accuracy,epoch)
                                writer.add_scalar('Accuracy/test',test_accuracy,epoch)
                                val_accuracies.append(val_accuracy)
                                train_accuracies.append(train_accuracy)
                                print("Epoch number:",epoch+1," Training accuracy:",train_accuracy,"Validation accuracy:",val_accuracy, "Testing accuracy:",test_accuracy)
                        #Keep track of best model and best parameters for reporting
                        if test_accuracy > best_accuracy:
                                best_accuracy = test_accuracy
                                best_params = {"n_layers":n_layers,"n_neurons":n_neurons,"use_dropout":use_dropout,
                                               "dropout_rate":dropout_rate,"learning_rate":learning_rate,"weight_decay":weight_decay}
                                best_model = model
                        else:
                                count +=1
                                if count >= early_stop:
                                        print("Best parameters:",best_params,"\n","Highest observed accuracy:",best_accuracy)
                                        return best_model
                print("Best parameters:",best_params,"\n","Highest observed accuracy:",best_accuracy)
                return best_model 
                        
def Predict(model,X, Normalize = False, level = 'All'):
        dataset_name = X
        data = pd.read_csv(dataset_name, sep=',', header=0) #Use input name +.csv to import data set to pandas dataframe. Pandas used for robustness of import function
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
                               
        x_test=np.array(transposed[1:])        #Select columns as features
        x_test = x_test.astype('Float64')
        if Normalize:
                x_train=normalize(x_train,axis=0,norm='l2')
                print('Normalizing!')
        x_test=torch.from_numpy(x_test)
        preds=[]
        for row in x_test:
                output=model(row.unsqueeze(dim=0).float())
                pred = output.data.max(1,keepdim=True)[1]
                preds.append(pred.item())



        return preds
if __name__=="__main__":
        par = read_params(sys.argv)
        model = Train_model(par['inp_f'],par['inp_m'],par['gridsearch'],par['threshold'],par['normalize'],par['feature_level'],par['epochs'],par['batch'],par['learning_rate'],par['dropout_rate'],par['early_stop'])
        torch.save(model.state_dict(), par['inp_f'].replace('.csv','')+'.pt')
        if (par['inp_u'] != None):
                print(Predict(model,par['inp_u'],Normalize=par['normalize'],level=par['feature_level']))
            
