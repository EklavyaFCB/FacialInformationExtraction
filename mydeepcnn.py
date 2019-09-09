import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

class MyDeepCNN:
	
	"""
	Convolutional neural network class which trains on data for facial tasks, 
	such as face recognition, or gender classification.
	
	Parameters
	----------
	x_train          : Train images data
	y_train          : Train images labels
	x_test           : Validation images data
	y_test           : Validation images labels
	x_test           : Test images data
	y_test           : Test images labels
	output_dir       : Directory for saving model
	lr               : Learning rate                           (default = 1e-3)
	beta_1           : Beta 1                                  (default = 0.9)
	beta_2           : Beta 2                                  (default = 0.999)
	nb_epochs        : Number of epochs                        (default = 50)
	epsilon          : Epsilon                                 (default = 1e-7)
	batch_size       : Train batch size                        (default = 64)
	seed             : Seed for random functions               (default = 0)
	num_features     : Number of features                      (default = 64)
	"""
	
	def __init__(self, 
				x_train,
				y_train,
				x_val,
				y_val,
				x_test,
				y_test,
				output_dir,
				lr=1e-3,
				beta_1=0.9,
				beta_2=0.999,
				nb_epochs=50,
				epsilon=1e-7,
				batch_size=64,
				seed=0,
				num_features=64):
		
		# Variables
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val
		self.x_test = x_test
		self.y_test = y_test

		self.output_dir = output_dir
		self.lr = lr
		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.nb_epochs = nb_epochs
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.seed = seed
		self.num_features = num_features

		self.model = Sequential()
		self.nb_classes = y_train.shape[1] # This assumes one-hot encode
		self.nb_images, self.edge, _, self.channels = self.x_train.shape
		self.preds_list_test = []
		self.probs_list_test = []
        
		self.history = None
		self.sess = tf.InteractiveSession()
		keras.backend.set_session(self.sess)

	def model_variables(self):
		'''Print out all class variables'''
		print('x_train      :', self.x_train.shape)
		print('y_train      :', self.y_train.shape)
		print('x_val        :', self.x_val.shape)
		print('y_val.       :', self.y_val.shape)
		print('x_test.      :', self.x_test.shape)
		print('y_test.      :', self.y_test.shape)
		print('output_dir   :', self.output_dir)
		print('lr           :', self.lr)
		print('beta_1       :', self.beta_1)
		print('beta_2       :', self.beta_2)
		print('nb_epochs    :', self.nb_epochs)
		print('epsilon      :', self.epsilon)
		print('batch_size   :', self.batch_size)
		print('seed         :', self.seed)
		print('num_features :', self.num_features)
		print('nb_classes   :', self.nb_classes)
		print('nb_images    :', self.nb_images)

	def create_model(self):
		'''Creates defined TensorFlow model'''
		self.model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', input_shape=(self.edge, self.edge, self.channels), kernel_regularizer=l2(0.01)))
		self.model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Conv2D(2*2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(2*2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
		self.model.add(BatchNormalization())
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		self.model.add(Dropout(0.5))
		
		self.model.add(Flatten())
		
		self.model.add(Dense(2*2*2*self.num_features, activation='relu'))
		self.model.add(Dropout(0.4))
		self.model.add(Dense(2*2*self.num_features, activation='relu'))
		self.model.add(Dropout(0.4))
		self.model.add(Dense(2*self.num_features, activation='relu'))
		self.model.add(Dropout(0.5))
		
		self.model.add(Dense(self.nb_classes, activation='softmax')) 

	def model_summary(self):
		'''Print model summary'''
		print(self.model.summary())

	def train(self, verbose=False):
		''' Trains the model on the parsed x_train and y_train values for the given number of epochs. Saves final .h5 model in output directory.'''
		# Convert True/False to 1/0 integer
		verbose = int(verbose)
        
		# Compile
		self.model.compile(loss=categorical_crossentropy,
								 optimizer=Adam(lr=self.lr, 
												beta_1=self.beta_1, 
												beta_2=self.beta_2, 
												epsilon=self.epsilon),
								 metrics=['accuracy'])
        
		# Train
		self.history = self.model.fit(np.array(self.x_train), np.array(self.y_train),
										 batch_size=self.batch_size,
										 epochs=self.nb_epochs,
										 verbose=verbose,
										 validation_data=(np.array(self.x_val), np.array(self.y_val)),
										 shuffle=True)

		# Save
		self.model.save(self.output_dir+'/my_keras_model.h5')
		self.model.save_weights(self.output_dir+'/my_keras_model_weights.h5')

	def test(self, verbose=False):
		'''Tests and prints the model's accuracy on the test data set.'''
		# Convert True/False to 1/0 integer
		verbose = int(verbose)

		# Reset
		self.preds_list_test = []
		self.probs_list_test = []

		# Run
		probs_test = self.model.predict(self.x_test, batch_size=self.batch_size, verbose=verbose)

		# Results
		self.probs_list_test.append(probs_test)
		self.preds_list_test = np.argmax(probs_test, axis=1)
		y_real_test = np.argmax(self.y_test, axis=1)

		# Print
		full_acc_test = np.mean(self.preds_list_test==y_real_test)
		print('Test accuracy achieved: %.3f' %full_acc_test)