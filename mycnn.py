import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from myutilitymethods import MyMethods

class MyCNN:
	
	"""
	Convolutional neural network class which trains on data for facial tasks, 
	such as face recognition, or gender classification.
	
	Parameters
	----------
	x_train          : Train images data
	y_train          : Train images labels
	x_test           : Test images data
	y_test           : Test images labels
	output_dir       : Directory for saving checkpoints
	num_to_class     : List for converting a number to a class value string
	class_to_num     : Dictionary for converting a class string to a number
	lr               : Learning rate                           (default = 1e-3)
	nb_epochs        : Number of epochs                        (default = 10)
	batch_size_train : Train batch size                        (default = 30)
	seed             : Seed for random functions               (default = 0)
	type             : Type of model                           (default = softmax)
	"""
    
	def __init__(self,
				 x_train,
				 y_train,
				 x_test,
				 y_test,
				 output_dir,
				 num_to_class,
				 class_to_num,
				 lr=1e-3,
				 nb_epochs=10,
				 batch_size_train=30,
				 seed=0,
				 final_activation='softmax'):

        # Variables
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.output_dir = output_dir
		self.lr = lr
		self.nb_epochs = nb_epochs
		self.batch_size_train = batch_size_train
		self.seed = seed
		self.final_activation = final_activation.lower()
		
		self.nb_classes = y_train.shape[1] # This assumes one-hot encode
		self.nb_images, self.edge, _, self.channels = x_train.shape
		self.nb_train_iterations = self.nb_images // batch_size_train
		self.im = tf.placeholder(tf.float32, [None, 28, 28, self.channels])
		self.labels = tf.placeholder(tf.float32, [None, self.nb_classes])
		
		self.accuracy_list = []
		self.losses_list = []
		self.test_losses_list = []
		self.test_accuracy_list = []
		self.val_accuracy_list = []
		self.val_losses_list = []
		self.preds_list = []
		self.probs_list = []
		self.preds_list_test = []
		self.probs_list_test = []
		
		self.mm = MyMethods(self.nb_classes, num_to_class, class_to_num)
		self.sess = tf.InteractiveSession()
		self.summary = tf.Summary()
		self.writer = tf.summary.FileWriter(self.output_dir)
		self.writer.add_graph(self.sess.graph)
        
	def model_variables(self):
		'''Print out all class variables'''
		print('x_train             :', self.x_train.shape)
		print('y_train             :', self.y_train.shape)
		print('x_test              :', self.x_test.shape)
		print('y_test              :', self.y_test.shape)
		print('output_dir          :', self.output_dir)
		print('lr                  :', self.lr)
		print('nb_epochs           :', self.nb_epochs)
		print('batch_size_train    :', self.batch_size_train)
		print('seed                :', self.seed)
		print('nb_classes          :', self.nb_classes)
		print('nb_images           :', self.nb_images)
		print('nb_train_iterations :', self.nb_train_iterations)
		print('im                  :', self.im)
		print('labels              :', self.labels)
        
	def set_up_saver(self):
		self.saver = tf.train.Saver()
	
	def create_model(self):
		'''Creates defined TensorFlow model'''
		with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
			self.first_layer = tf.layers.conv2d(self.im, 32, (4,4), strides=1, activation=tf.nn.relu)            
			self.second_layer = tf.layers.conv2d(self.first_layer, 64, (4,4), strides=2, activation=tf.nn.relu)
			self.third_layer = tf.layers.conv2d(self.second_layer, 128, (4,4), strides=2, activation=tf.nn.relu)
			self.fourth_layer = tf.layers.conv2d(self.third_layer, 256, (4,4), strides=2, activation=None)
			self.flattened = tf.layers.flatten(self.fourth_layer)
			self.logits = tf.layers.dense(self.flattened, self.nb_classes, activation=None) 
			if self.final_activation == 'softmax':
				self.preds = tf.nn.softmax(self.logits)
			elif self.final_activation == 'sigmoid':
				self.preds = tf.nn.sigmoid(self.logits)
			else:
				print("Error: the model.final_activation parameter can only either be 'softmax' or 'sigmoid'. ")

	def model_summary(self):
		'''Prints out the model's architecture shapes'''
		print('first_layer  : conv2D  -', self.first_layer.shape)
		print('second_layer : conv2D  -', self.second_layer.shape)
		print('third_layer  : conv2D  -', self.third_layer.shape)
		print('fourth_layer : conv2D  -', self.fourth_layer.shape)
		print('flattened.   : Flatten -', self.flattened.shape)
		print('logits       : Dense   -', self.logits.shape)
		print('preds        : Softmax -', self.preds.shape)
	
	def compute_loss(self):
		'''Computes loss for the model'''
		with tf.variable_scope('loss'):
			if self.final_activation == 'softmax':
				self.loss = tf.losses.softmax_cross_entropy(self.labels, logits=self.logits)
				self.loss_summ = tf.summary.scalar("softmax_loss", self.loss)
			elif self.final_activation == 'sigmoid':
				self.loss = tf.losses.sigmoid_cross_entropy(self.labels, logits=self.logits)
				self.loss_summ = tf.summary.scalar("sigmoid_loss", self.loss)
			else:
				print("Error: the model.final_activation parameter can only either be 'softmax' or 'sigmoid'. ")
	
	def optimizer(self):
		'''Defines optimizer used for the model'''
		with tf.variable_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
			self.model_vars = tf.trainable_variables()
			self.trainer = optimizer.minimize(self.loss, var_list=self.model_vars)
	 
	def train(self, verbose=False, cross_k_fold_validation=True):
		'''
		Trains the model on the parsed x_train and y_train values for the given number of epochs. 
		Saves checkpoints in output directory.
		'''
		
		if cross_k_fold_validation:
			# We split the train data into k equal subsets for k-fold cross validation
			# Note: these subsets are different from iteration batches used for optimisated gradient descent
			cut_x_train = self.x_train.reshape(self.x_train.shape[0], np.prod(self.x_train.shape[1:])) # (20000, 2352)
			subset_size = cut_x_train.shape[0]//self.nb_epochs
			cut_x_train = cut_x_train.reshape(self.nb_epochs, cut_x_train.shape[1]*subset_size) # (200, 235200)
			
			cut_y_train = np.argmax(self.y_train, axis=1) # (20000,)
			cut_y_train = cut_y_train.reshape(self.nb_epochs, subset_size) # (200, 100)
			
		else:
			self.x_train, self.y_train, self.x_val, self.y_val = self.mm.split_train_test(self.x_train, self.y_train)
			
			x_in = self.x_train
			y_in = self.y_train
			
			if self.y_val.ndim == 1:
				self.y_val = self.mm.one_hot_encode(self.y_val)
		
		for epoch in range(self.nb_epochs):
			
			if cross_k_fold_validation:
				# Get validation data from the training set, both which change at each epoch
				x_in = np.delete(cut_x_train, epoch, axis=0) # (199, 235200)
				y_in = np.delete(cut_y_train, epoch, axis=0) # (199, 100)
				self.x_val = cut_x_train[epoch] # (1, 235200) = (235200,)
				self.y_val = cut_y_train[epoch] # (1, 100) = (100,)
				
				# Reshape back to standard
				x_in = x_in.reshape(x_in.shape[0]*subset_size, x_in.shape[1]//subset_size) # (19900, 2352)
				x_in = x_in.reshape(x_in.shape[0], 28, 28, self.channels) # (19900, 28, 28, 3)
				
				y_in = y_in.reshape(y_in.shape[0]*subset_size) # (19900,)
				if y_in.ndim == 1:
					y_in = self.mm.one_hot_encode(y_in) # (19900, 4)
				
				self.x_val = self.x_val.reshape(subset_size, 28, 28, self.channels) # (100, 28, 28, 3)
				if self.y_val.ndim == 1:
					self.y_val = self.mm.one_hot_encode(self.y_val) # (100, 4)
			
			# Reset
			self.preds_list = []
			self.probs_list = []
			self.probs_list_val = []
			self.preds_list_val = []
			
			# Shuffle order for both subsets
			x_in, y_in = shuffle(x_in, y_in, random_state=self.seed)
			self.x_val, self.y_val = shuffle(self.x_val, self.y_val, random_state=self.seed)
			
			# Loop through iterations
			for i in range(self.nb_train_iterations):
				
				# Take batch 
				input_x_train = x_in[i*self.batch_size_train: (i+1)*self.batch_size_train]
				input_y_train = y_in[i*self.batch_size_train: (i+1)*self.batch_size_train]
				
				# Train model
				_ , probs, loss, loss_summ = self.sess.run([self.trainer, self.preds, self.loss, self.loss_summ], 
                                                          feed_dict={self.im: input_x_train, self.labels: input_y_train})
				
				self.writer.add_summary(loss_summ, epoch * self.nb_train_iterations + i)
			
			# We want the entire train set score after each epoch for those epoch values
			probs, loss = self.sess.run([self.preds, self.loss], feed_dict={self.im: x_in, self.labels: y_in})
			y_hat = np.argmax(probs, axis=1)
			
			self.probs_list.append(probs)
			self.preds_list.append(y_hat)

			y_real = np.argmax(y_in, axis=1)
			acc_train = np.mean(self.preds_list == y_real)
			self.accuracy_list.append(acc_train)
			self.losses_list.append(loss) 
			
			# We want the entire validation set score after each epoch
			probs_val, loss_val = self.sess.run([self.preds, self.loss], feed_dict={self.im: self.x_val, self.labels: self.y_val})
			
			y_hat_val = np.argmax(probs_val, axis=1)
			y_real_val = np.argmax(self.y_val, axis=1)
			
			self.probs_list_val.append(probs_val)
			self.preds_list_val.append(y_hat_val)
			
			acc_val = np.mean(y_hat_val == y_real_val)
			self.val_accuracy_list.append(acc_val)
			self.val_losses_list.append(loss_val)
			
			# Print
			if verbose:
				print(f'Epoch {epoch}, Train acc {np.round(acc_train, 2)}, Validation acc {np.round(acc_val, 2)}')
			
			# Save
			self.saver.save(self.sess, self.output_dir, global_step=epoch)
			
		# For the final epoch we want the predictions for the non-shuffled x_train and y_train (not x_in and y_in)
		probs = self.sess.run(self.preds, feed_dict={self.im: self.x_train, self.labels: self.y_train})
		self.probs_list = probs
		self.preds_list = np.argmax(probs, axis=1)
		
		# Flatten
		self.probs_list_val = np.concatenate(self.probs_list_val, axis=0)
		self.preds_list_val = np.concatenate(self.preds_list_val, axis=0)
			
	def test(self):
		'''Tests and prints the model's accuracy on the test data set or a custom input.'''
        
        # Reset
		self.preds_list_test = []
		self.probs_list_test = []
		
		# Run
		probs_test = self.sess.run(self.preds, feed_dict={self.im: self.x_test, self.labels: self.y_test})
		
		# Append results
		self.probs_list_test.append(probs_test)
		self.preds_list_test.append(np.argmax(probs_test, axis=1))
		
		# Flatten
		self.probs_list_test = np.concatenate(self.probs_list_test, axis=0)
		self.preds_list_test = np.concatenate(self.preds_list_test, axis=0)
		
		# Print final accuracy
		y_real_test = np.argmax(self.y_test, axis=1)
		full_acc_test = np.mean(self.preds_list_test==y_real_test)
		print('Test accuracy achieved: %.3f' %full_acc_test)