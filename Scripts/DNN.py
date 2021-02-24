import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from datetime import datetime

# He et al. initialization from https://arxiv.org/abs/1502.01852
he_init = tf.contrib.layers.variance_scaling_initializer()

# This class inherits from Sklearn's BaseEstimator and ClassifierMixin 
class DNNClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, n_hidden_layers=10, n_neurons=25, optimizer_class=tf.train.AdamOptimizer, learning_rate=0.01, batch_size=40, activation=tf.nn.elu, initializer=he_init, batch_norm_momentum=None, dropout_rate=0.2, max_checks_without_progress=20,show_progress=10, tensorboard_logdir=None, random_state=None):
		
		##Initialize the class with the default hyperparameters as determined from other runs
		self.n_hidden_layers = n_hidden_layers
		self.n_neurons = n_neurons
		self.optimizer_class = optimizer_class
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.activation = activation
		self.initializer = initializer
		self.batch_norm_momentum = batch_norm_momentum
		self.dropout_rate = dropout_rate
		self.max_checks_without_progress = max_checks_without_progress
		self.show_progress = show_progress
		self.random_state = random_state
		self.tensorboard_logdir = tensorboard_logdir
		self._session = None #Instance variables preceded by _ are private members
		
	def _dnn(self, inputs):
		'''This method builds the hidden layers and
		 Provides for implementation of batch normalization and dropout'''

		for layer in range(self.n_hidden_layers):

			# Apply dropout if specified
			if self.dropout_rate:
				inputs = tf.layers.dropout(inputs, rate=self.dropout_rate, training=self._training)
			# Create the hidden layer
			inputs = tf.layers.dense(inputs, self.n_neurons, 
									 activation=self.activation, 
									 kernel_initializer=self.initializer, 
									 name = "hidden{}".format(layer+1))

			# Apply batch normalization if specified
			if self.batch_norm_momentum:
				inputs = tf.layers.batch_normalization(inputs,momentum=self.batch_norm_momentum,
													training=self._training)
				
			# Apply activation function
			inputs = self.activation(inputs, name="hidden{}_out".format(layer+1))
		return inputs

        ##Code for graph was mostly derived from online tutorial for tensorboard
		
	def _construct_graph(self, n_inputs, n_outputs):
		'''This method builds the complete Tensorflow computation graph
			n_inputs: number of features 
			n_outputs: number of classes
		'''

		if self.random_state:
			tf.set_random_seed(self.random_state)
			np.random.seed(self.random_state)
		 
		# Placeholders for training data, labels are class exclusive integers
		X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
		y = tf.placeholder(tf.int32, shape=[None], name="y")
		
		# Create a training placeholder 
		if self.batch_norm_momentum or self.dropout_rate:
			self._training = tf.placeholder_with_default(False, shape=[], name="training")
		else:
			self._training = None
		
		# Output after hidden layers 
		pre_output = self._dnn(X)
		
		# Outputs from output layer
		logits = tf.layers.dense(pre_output, n_outputs, kernel_initializer=he_init, name="logits")
		probabilities = tf.nn.softmax(logits, name="probabilities")
		
		''' Cost function is cross entropy and loss is average cross entropy. Sparse softmax must be used because shape of logits is [None, n_classes] and shape of labels is [None]'''
		xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
		loss = tf.reduce_mean(xentropy, name="loss")
		
		'''Optimizer and training operation. The control dependency is necessary for implementing batch normalization. The training operation must be dependent on the batch normalization.'''

		optimizer = self.optimizer_class(learning_rate=self.learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			training_op = optimizer.minimize(loss)
		
		# Metrics for evaluation
		correct = tf.nn.in_top_k(logits, y, 1)    
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32),name="accuracy") #Accuracy used due to balanced datasets. Implementation of different metrics will be useful for broader application

		# Initializer and saver 
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
	
		if self.tensorboard_logdir:
			now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
			tb_logdir = self.tensorboard_logdir + "/run-{}".format(now)
			cost_summary = tf.summary.scalar("validation_loss", loss)
			acc_summary = tf.summary.scalar("validation_accuracy", accuracy)
			merged_summary = tf.summary.merge_all()
			file_writer = tf.summary.FileWriter(tb_logdir,tf.get_default_graph())
			
			self._merged_summary = merged_summary
			self._file_writer = file_writer
		
		self._X, self._y = X, y
		self._logits = logits
		self._probabilities = probabilities
		self._loss = loss
		self._training_op = training_op
		self._accuracy = accuracy
		self._init, self._saver = init, saver
		
		
	def close_session(self):
		if self._session:
			self._session.close()
			
	def _get_model_parameters(self):
		# Retrieves the value of all the variables in the network 
		with self._graph.as_default():
			gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		return {gvar.op.name: value for gvar, value in 
				zip(gvars, self._session.run(gvars))}
	#Especially important for use of grid search to determine best default parameters to use
	def _restore_model_parameters(self, model_params):
		# Restores the value of all variables using tf assign operations
		# First retrieve the list of all the graph variables
		gvar_names = list(model_params.keys())
		
		# Then retrieve all the assignment operations in the graph
		assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}

		# Fetch the initialization values of the assignment operations
		'''graph.get_operation_by_name(operation).inputs returns the input to the given operation; because these are all assignment operations, the second argument to inputs is the value assigned to the variable'''
		init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
		# Create a dictionary mapping initial values to values after training
		feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
		# Assign the trained value to all the variables in the graph
		self._session.run(assign_ops, feed_dict=feed_dict)
		
	def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
		# Method to train the model. Implements early stopping if validation data is provided 
		
		self.close_session()
		n_inputs = X.shape[1] # Number of features
		
		# If labels are provided in one_hot form, convert to integer class labels
		y = np.array(y)
		y_valid = np.array(y_valid)
		
		if len(y.shape) == 2:
			y = np.argmax(y, axis=1)
	 
		if len(y_valid.shape) == 2:
			y_valid = np.argmax(y_valid, axis=1)

		self.classes_ = np.unique(y)
		n_outputs = len(self.classes_) # Number of classes
	
		# Tensorflow expects labels from 0 to n_classes - 1. 
		self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
		labels = [self.class_to_index_[label] for label in y]
		y = np.array(labels, dtype=np.int32)
		
		self._graph = tf.Graph()
			
		# Build the computation graph with self as default graph
		with self._graph.as_default():
			self._construct_graph(n_inputs, n_outputs)
			
		# Early stopping parameters
		checks_without_progress = 0 
		best_loss = np.float("inf")
		best_parameters = None
		
		self._session = tf.Session(graph=self._graph)

		with self._session.as_default() as sess:
			# Initialize all variables
			self._init.run()
			num_instances = X.shape[0] # Total number of training instances
			for epoch in range(n_epochs):
				rnd_idx = np.random.permutation(num_instances)
				for rnd_indices in np.array_split(rnd_idx, num_instances // self.batch_size): #Batch implementation
					X_batch, y_batch = X[rnd_indices], y[rnd_indices]
					feed_dict = {self._X: X_batch, self._y: y_batch}
					if self._training is not None:
						feed_dict[self._training] = True
					train_acc, _ = sess.run([self._accuracy,self._training_op],feed_dict) #Calculate training accuracy

				# Early stopping implementation
				if X_valid is not None and y_valid is not None:
					feed_dict_valid = {self._X: X_valid, self._y: y_valid}

					# Write summary for tensorboard
					if self.tensorboard_logdir:
						val_acc, val_loss, summary = sess.run([self._accuracy, self._loss, self._merged_summary], feed_dict=feed_dict_valid)

						self._file_writer.add_summary(summary, epoch)

					else:
						val_acc, val_loss = sess.run([self._accuracy, self._loss], feed_dict=feed_dict_valid)

					###Still need to figure out how to call logging for gridsearch function, if possible
					
					# Show training progress every show_progress epochs
					if self.show_progress:
						if epoch % self.show_progress == 0:
							print("Epoch: {} Current training accuracy: {:.4f} Validation Accuracy: {:.4f} Validation Loss {:.6f}".format(
								epoch+1, train_acc, val_acc, val_loss))

					# Check to see if model is improving 
					if val_loss < best_loss:
						best_loss = val_loss
						checks_without_progress = 0
						best_parameters = self._get_model_parameters()
					else:
						checks_without_progress += 1

					if checks_without_progress > self.max_checks_without_progress:
						print("Stopping Early! Loss has not improved in {} epochs".format(
										   self.max_checks_without_progress))
						break
			   
				# No validation set provided
				else:
					if self.show_progress:
						if epoch % self.show_progress == 0:
							print("Epoch: {} Current training accuracy: {:.4f}".format(
								epoch+1, train_acc))
						
			# In the case of early stopping, restore the best weight values
			if best_parameters:
				self._restore_model_parameters(best_parameters)
				return self
			
	def predict_probabilities(self, X):
		# Predict the probabilities of each class 
		if not self._session:
			raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
		with self._session.as_default() as sess:
			return self._probabilities.eval(feed_dict={self._X: X})

	def predict(self, X):
		# Predict the classes themselves and return with shape=(None,)
		class_indices = np.argmax(self.predict_probabilities(X), axis=1)
		predictions = np.array([[self.classes_[class_index]] for class_index in class_indices], dtype=np.int32)
		return np.reshape(predictions, (-1,))
		
	def save(self, path):
		# Save the model to provided path
		self._saver.save(self._session, path)


##Much guidance on the classifier was obtained from Will Koerhsen's blog at https://medium.com/@williamkoehrsen/deep-neural-network-classifier-32c12ff46b6c . Tweaks to the code for this implementation were made, but most of the code was similar due to the simplicity of the tesnorflow DNN package.






##Function to run desired dataset

def RunAnalysis(GridSearch=False, Threshold = 0, Normalize = False, level='All'):

        dataset_name = input('Enter the name of the taxonomic profile to analyze in .csv format:')      #Only the name of the csv files are needed. Do not add .csv
        metadata_name = input('Enter the name of the metadata file for the dataset in .csv format: ')
        
        ##Initialize classifier

        dnn = DNNClassifier(tensorboard_logdir="/tensorboard_stats", random_state=None)

        ##Data processing

        data = pd.read_csv(dataset_name+'.csv', sep=',', header=0) #Use input name +.csv to import data set to pandas dataframe. Pandas used for robustness of import function
        my_data = data.to_numpy() #Convert to numpy array
        transposed = my_data.transpose()#Transpose Numpy array for easier manipulation and feature extraction
        remove_indeces = []
        if level == 'Species':
                for x in range(len(transposed[0])):
                       if not 's_' in str(transposed[0][x]):
                               remove_indeces.append(int(x))
                transposed = np.delete(transposed, remove_indeces, 1)

        elif level == 'Genus':
                for x in range(len(transposed[0])):
                       if not 'g_' in str(transposed[0][x]):
                               remove_indeces.append(int(x))
                transposed = np.delete(transposed, remove_indeces, 1)
                               
        x_train=np.array(transposed[1:])        #Select columns as features
        x_train = x_train.astype('Float64')
        if Normalize:
                x_train = tf.keras.utils.normalize(x_train, 0, 2)
                print('Normalizing!')
        labels = pd.read_csv(metadata_name+'.csv', sep=',', header=0)        #Import metadata file
        labels = labels.to_numpy()      #Convert pandas df to numpy for easier handling
        labels=labels.transpose()       #Transpose array for easier manipulation


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
        
        ##Fit an initial model to the data using default parameters

        dnn.fit(x_train, y_train,100,x_val, y_val)

        
        ##Randomized grid search implementation if Gridsearch=True
        if GridSearch:                                                                                                  #Only enter block if GridSearch is set to True for the RunAnalysis function
                parameter_tunes = {
                        'n_hidden_layers' :  [5,10,15,20],
                        'n_neurons' :    [25,50,75,100],
                        'learning_rate' :       [0.0001, 0.001, 0.01, 0.1, 1, 10],
                        'batch_size' :  [20,30,40,50,60,70,80],
                        'activation' :  [tf.nn.elu,tf.nn.relu,tf.nn.relu6,tf.nn.selu,tf.nn.sigmoid,tf.nn.tanh],
                        'dropout_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                        'optimizer_class': [tf.train.AdamOptimizer, tf.train.AdadeltaOptimizer, tf.train.AdagradDAOptimizer, tf.train.AdagradOptimizer],
                        'max_checks_without_progress':[10,20,30,40,50,100]
                        }
                RGS = sklearn.model_selection.RandomizedSearchCV(DNNClassifier(random_state=42), parameter_tunes)       #Register gridsearch class as a variable for further analysis
                Search_results = RGS.fit(x_train, y_train)                                                              #Train the model using a randomized grid search
                                                                                            
                print('Grid search accuracy:',(accuracy_score(y_test,Search_results.predict(x_test))),'\n ','Default parameter accuracy:',(accuracy_score(y_test,dnn.predict(x_test)))) #Output the testing accuracy of the grid search model and the default parameter model. Useful for comparison
                return Search_results.best_params_                                                                      #Output the best model parameters for easy default parameter tuning.
        else:                                                                                                           #If GidSearch=False, return testing accuracy using default parameters. 
                print('Default parameter accuracy: ', (accuracy_score(y_test,dnn.predict(x_test))))
                return 
