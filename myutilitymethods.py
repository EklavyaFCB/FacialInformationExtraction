import os
import sys
import cv2
import time
import keras
import random
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy import interp
import albumentations as A
from itertools import cycle
from IPython import display
from sklearn import metrics
from datetime import datetime
#from mtcnn.mtcnn import MTCNN
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix as sk_cm

class MyMethods:

	def __init__(self, NUM_CLASSES, num_to_class, class_to_num, class_colors=['C1', 'C2', 'C3', 'C0']):
		'''Initialise with NUM_CLASSES, num_to_class, class_to_num'''
		self.NUM_CLASSES = NUM_CLASSES
		self.num_to_class = num_to_class
		self.class_to_num = class_to_num
		if self.NUM_CLASSES != 4:
			class_colors = self.randomColorGenerator(self.NUM_CLASSES)
		self.class_colors = class_colors
        
	def read_csv(filepath, aug_factor=10, print_progress=False, img_size=28):
		'''Takes in .csv and returns the compressed images's pixels in RGB format as Numpy array'''
		img = pd.read_csv(filepath, header=None).values
		return img.reshape(img.shape[0], img_size, img_size, 3)

	def equalise_image(self, image, eq_type='HSV'):
		'''Return HSV or YCrCb equalised image'''
		if eq_type == 'HSV':
			H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
			eq_V = cv2.equalizeHist(V)
			return cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
		elif eq_type == 'YCrCb':
			Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)) 
			eq_Y = cv2.equalizeHist(Y)
			return cv2.cvtColor(cv2.merge([eq_Y, Cr, Cb]), cv2.COLOR_YCrCb2RGB)

	def equalise_gray_image(self, image, eq_type='HSV'):
		'''Return HSV or YCrCb equalised image'''
		if eq_type == 'HSV':
			H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_GRAY))
			eq_V = cv2.equalizeHist(V)
			return cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
		elif eq_type == 'YCrCb':
			Y, Cr, Cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)) 
			eq_Y = cv2.equalizeHist(Y)
			return cv2.cvtColor(cv2.merge([eq_Y, Cr, Cb]), cv2.COLOR_YCrCb2RGB)
	
	def convertToRGB(self, image):
		'''Convert image from BGR to RGB'''
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	def resize_image(self, img, img_size=28):
		'''Resize image to chosen size'''
		return cv2.resize(img, (img_size, img_size))
	
	def standardise_images(self, images):
		'''Standardise a batch of images'''
		std_images = []
		for img in images:
			std_images.append(self.standardise_image(img))
		return std_images
	
	def standardise_image(self, image):
		'''Standardise an image per channel'''
		means = image.mean(axis=(0,1))
		stds = image.std(axis=(0,1))
		return (image - means) / stds
	
	def normalise_image(self, image, norm_range=(0,1)):
		'''Normalise an image per channel'''
		left_term = norm_range[1]-norm_range[0]
		middle_term = (image-image.min(axis=(0,1))) / (image.max(axis=(0,1)) - image.min(axis=(	0,	1)))
		right_term = norm_range[0]
		return left_term * middle_term + right_term
	
	def greyscale_image(self, image):
		'''Greyscale an RGB image'''
		return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	
	def print_stats(self, mypic):
		'''Print min, max, mean, std, and variance of a given image'''
		print('Min :', np.round(mypic.min(axis=(0,1)),  2))
		print('Max :', np.round(mypic.max(axis=(0,1)),  2))
		print('Mean:', np.round(mypic.mean(axis=(0,1)), 2)) 
		print('Std :', np.round(mypic.std(axis=(0,1)),  2))
		print('Var :', np.round(mypic.var(axis=(0,1)),  2))

	def describe_data(self, x_train, y_train, x_test, y_test):
		'''Describe train and test data'''
		print('Shapes')
		print('x_train :', x_train.shape)
		print('y_train :', y_train.shape)
		print('x_test  :', x_test.shape)
		print('y_test  :', y_test.shape)
		print('')
		print('Min')
		print('x_train :', np.round(x_train.min(),3))
		print('x_test  :', np.round(x_test.min(), 3))
		print('')
		print('Max')
		print('x_train :', np.round(x_train.max(),3))
		print('x_test  :', np.round(x_test.max(), 3))
		print('')
		print('Means')
		print('x_train :', np.round(x_train.mean(),3))
		print('x_test  :', np.round(x_test.mean(), 3))
		print('')
		print('Std')
		print('x_train :', np.round(x_train.std(),3))
		print('x_test  :', np.round(x_test.std(), 3))
	
	
	def one_hot_encode(self, y):
		'''One-hot encode 1D list of labels'''
		return np.eye(self.NUM_CLASSES)[y]
	
	def split_train_test(self, x, y, split=0.9):
		'''Split dataset into train and test sets'''
		x_train = x[:int(len(x)*split)]
		y_train = y[:int(len(y)*split)]
		x_test = x[int(len(x)*split):]
		y_test = y[int(len(y)*split):]
		return x_train, y_train, x_test, y_test
	
	def split_train_val_test(self, x, y, split1=0.9, split2=0.95):
		'''Split dataset into train, validation, and test sets'''
		x_train, x_val, x_test = np.split(x, [int(len(x)*split1), int(len(x)*split2)])
		y_train, y_val, y_test = np.split(y, [int(len(y)*split1), int(len(y)*split2)])
		return x_train, y_train, x_val, y_val, x_test, y_test
	
	def compute_rolling_average(self, myList):
		'''Computes and returns the rolling average of a given list'''
		rolling_avg = []
		for i, v in enumerate(myList, 1):
			rolling_avg.append(np.sum(myList[:i])/i)
		return rolling_avg
	
	def plot_results(self, x_test, y_real, y_hat, probs, sleep_time=3.0, dynamic=True, include_title=True):
		'''Plot image and probability distribution bar plot of faces'''
		for i, pic in enumerate(x_test):
			fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(5,2.5), constrained_layout=True, dpi=	100)
			if include_title:
				fig.suptitle(f'Actual:{self.num_to_class[y_real[i]]}, Predicted:{self.num_to_class[y_hat[i]	]}', fontsize=15)
			# Left
			ax1.imshow(cv2.resize(x_test[i], (x_test[i].shape[0], x_test[i].shape[0])))
			ax1.axes.get_xaxis().set_visible(False) 
			ax1.axes.get_yaxis().set_visible(False) 
			# Right
			ax2.barh(np.arange(self.NUM_CLASSES), probs[i])
			ax2.set_yticks(np.arange(self.NUM_CLASSES))
			ax2.set_yticklabels(self.num_to_class)
			ax2.invert_yaxis()
			ax2.set_xlim(left=0.0, right=1)
			# Method 1:
			# Plot
			#plt.show()
			# GIF
			#time.sleep(sleep_time)
			#display.clear_output(wait=True)
			# Method 2
			if dynamic:
				display.clear_output(wait=True)
				display.display(plt.gcf())
				time.sleep(sleep_time)
			else:
				plt.show()

	def plot_metrics(self, accuracy, loss, val_accuracy=None, val_loss=None, dpi=100, save_title=None):
		'''Plot graph subplots of the accuracy and loss of the model over the epochs during training.'''
		fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,5), constrained_layout=True, dpi=	dpi)
		fig.suptitle('Metrics', fontsize=15)
		# Plot
		axes[0,0].plot(accuracy, label='Train Accuracy')
		axes[1,0].plot(loss, label='Train Loss')
		axes[0,1].plot(self.compute_rolling_average(accuracy), label='Average Train Accuracy')
		axes[1,1].plot(self.compute_rolling_average(loss), label='Average Train Loss')
		if val_accuracy is not None and val_loss is not None:
			axes[0,0].plot(val_accuracy, label='Validation Accuracy')
			axes[1,0].plot(val_loss, label='Validation Loss')
			axes[0,1].plot(self.compute_rolling_average(val_accuracy), label='Average Validation 	Accuracy')
			axes[1,1].plot(self.compute_rolling_average(val_loss), label='Average Validation Loss')
		# Titles
		axes[0,0].set_title('Accuracy')
		axes[1,0].set_title('Loss')
		axes[0,1].set_title('Rolling Accuracy Average')
		axes[1,1].set_title('Rolling Loss Average')
		# Axis
		#axes[0,0].set_xlabel('Iterations')
		#axes[0,0].set_ylabel('Accuracy')
		#axes[1,0].set_xlabel('Loss')
		#axes[0,1].set_title('Rolling Accuracy Average')
		#axes[1,1].set_title('Rolling Loss Average')
		# Limits
		axes[0,0].set_ylim(bottom=0.0, top=1)
		axes[0,1].set_ylim(bottom=0.0, top=1)
		# Plot
		if val_accuracy is not None and val_loss is not None:
			plt.legend()
		if save_title is not None:
			fig.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)
		plt.show()
	
	
	def plot_pca(self, x_train, y_train, dpi=150, title=None):
		'''Plot PCA plot of training set'''
		n_train = x_train.shape[0]
		nb_features = np.prod(x_train.shape[1:])
		x_train = x_train.reshape((n_train, nb_features))
		# PCA
		pca = PCA(n_components=2)
		x_fit = pca.fit_transform(x_train)
		# Plot
		plt.figure(figsize=(8,5), dpi=dpi) 
		for num in set(y_train):
			#plt.scatter(x_fit[y_train==num, 0], x_fit[y_train==num, 1], label=num_to_class[num], 	color=class_colors[num], s=1)
			plt.scatter(x_fit[y_train==num, 0], x_fit[y_train==num, 1], s=1)
		if title:
			plt.title(title)
		plt.tight_layout()
		if title is not None: 
			plt.savefig(f'{title}.pdf', bbox_inches='tight', format='pdf', dpi=200)
		plt.show()
	
	def get_false_classifications(self, y_real, all_preds):
		'''Get false classification values'''
		return np.where((y_real == all_preds)*1 == 0)[0]
	
	def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, cmap='GnBu', 	dpi=100):
		'''Plot confusion matrix with y_real, y_hat, and classes.'''
		if not title:
				if normalize:
					title = 'Normalized confusion matrix'
				else:
					title = 'Confusion matrix, without normalization'
		# Compute confusion matrix
		cm = sk_cm(y_true, y_pred)
		# Only use the labels that appear in the data
		classes = classes[unique_labels(y_true, y_pred)]
		if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		# Subplots
		fig, ax = plt.subplots(dpi=dpi)
		im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.figure.colorbar(im, ax=ax)
		# Label
		ax.set(xticks=np.arange(cm.shape[1]),
					 yticks=np.arange(cm.shape[0]),
					 xticklabels=classes, 
					 yticklabels=classes,
					 title=title,
					 ylabel='True label',
					 xlabel='Predicted label')
		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_yticklabels(), rotation=90, ha="right", rotation_mode="anchor")
		# Loop over data dimensions and create text annotations.
		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		# Loop
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				ax.text(j, i, format(cm[i, j], fmt), 
								ha="center", va="center", 
								color="white" if cm[i, j] > thresh else "black")
		# Plot
		fig.tight_layout()
		plt.show()

	def plot_all_confusion_matrices(y_true, y_pred, y_true_val, y_pred_val, y_true_test, y_pred_test, classes, normalize=False, title=None, cmap='GnBu', dpi=150, save_title=None):
		'''Plot train, validation, and test confusion matrices'''
		if not title:
			if normalize:
				title = 'Normalized Confusion Matrices'
			else:
				title = 'Non-Normalized Confusion Matrices'
		# Compute confusion matrix
		cm_train = sk_cm(y_true, y_pred)
		cm_val = sk_cm(y_true_val, y_pred_val)
		cm_test = sk_cm(y_true_test, y_pred_test)
		# Only use the labels that appear in the data
		classes = classes[unique_labels(y_true, y_pred)]
		if normalize: 
			cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
			cm_val   = cm_val.astype('float')   / cm_val.sum(axis=1)  [:, np.newaxis]
			cm_test  = cm_test.astype('float')  / cm_test.sum(axis=1) [:, np.newaxis]
        # Lists
		cms = [cm_train, cm_val, cm_test]
		titles = ['Train', 'Validation', 'Test']
        # Plot
		fig, axes = plt.subplots(nrows=1, ncols=3, dpi=dpi, figsize=(15, 8))
		#fig.suptitle(title)
		for i, ax in enumerate(axes):
			im = ax.imshow(cms[i], interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
			# Label
			ax.set(xticks=np.arange(cms[i].shape[1]),
					yticks=np.arange(cms[i].shape[0]),
					xticklabels=classes, 
					yticklabels=classes,
					title=titles[i])
			# Rotate the tick labels and set their alignment.
			plt.setp(ax.get_yticklabels(), rotation=90, ha="right", rotation_mode="anchor")
		# Loop
		for c, cm in enumerate(cms):
				# Loop over data dimensions and create text annotations.
				fmt = '.2f' if normalize else 'd'
				thresh = cm.max() / 2.
				for i in range(cm.shape[0]):
						for j in range(cm.shape[1]):
								axes[c].text(j, i, format(cm[i, j], fmt), 
														 ha="center", va="center", 
														 color="white" if cm[i, j] > thresh else "black")
		# For only one ax
		axes[0].set(ylabel='True label',)
		axes[1].set(xlabel='Predicted label')
		# Adjust
		fig.subplots_adjust(right=0.8)
		cbar_ax = fig.add_axes([0.83, 0.315, 0.025, 0.375]) # [left, bottom, width, height]
		fig.colorbar(im, cax=cbar_ax)
		if save_title is not None:
			fig.savefig(f'{save_title}.pdf', format='pdf', dpi=200)
		# Plot
		plt.show()

	def weak_aug(self, p=0.5):
		'''Create a weakly augmented image framework'''
		return A.Compose([
            A.HorizontalFlip(),
            A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise(),], p=0.2),
            A.OneOf([A.MotionBlur(p=0.2), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1),], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
            A.OpticalDistortion(p=0.2),
            A.OneOf([A.CLAHE(clip_limit=2), A.IAASharpen(), A.IAAEmboss(),], p=0.3),
        ], p=p)

	def strong_aug(self, p=0.5):
		'''Create a strongly augmented image framework'''
		return A.Compose([
            #RandomRotate90(), #Flip(), #Transpose(),
            A.HorizontalFlip(),
            A.OneOf([A.IAAAdditiveGaussianNoise(), A.GaussNoise(),], p=0.2),
            A.OneOf([A.MotionBlur(p=0.2), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1),], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([A.OpticalDistortion(p=0.2), A.GridDistortion(p=0.1), A.IAAPiecewiseAffine(p=0.2),], p=0.2),
            A.OneOf([A.CLAHE(clip_limit=2), A.IAASharpen(), A.IAAEmboss(), A.RandomBrightnessContrast(),], p=0.3),
            A.HueSaturationValue(p=0.3),
        ], p=p)
	

	def augment_images(self, img, aug_factor):
		'''Augment and return images'''
		augmented_images = []
		my_weak_aug = self.weak_aug(p=1)
		my_strong_aug = self.strong_aug(p=1)
		
		# Augment and standardise
		if aug_factor > 1:
			for i in range(aug_factor):
				if i < int(aug_factor*0.7):
					augmented_images.append(my_weak_aug(image=img)['image'])
				else:
					augmented_images.append(my_strong_aug(image=img)['image'])
			
			augmented_images = np.array(augmented_images)
			return np.concatenate((img[None,:,:,:], augmented_images))
		else:
			return img
	
	def process_data(self, raw_data_list, augs):
		'''Convert images to RGB, equalise, augment and standardise.'''
		processed_data = []
		batch = []
		for i, raw_data in enumerate(raw_data_list):
			for img in raw_data:
				img = self.convertToRGB(img.astype('uint8'))
				img = self.equalise_image(img, eq_type='HSV')
				imgs = self.augment_images(img, augs[i])
				imgs = self.standardise_images(imgs)
				batch.append(imgs)
				processed_data.append(np.concatenate(batch))
				batch = []
		return processed_data
    
	def make_fpr_tpr_auc_dicts(self, y, probs_list):
		'''Compute and return the ROC curve and ROC area for each class in dictionaries'''
		# Dicts
		fpr = dict()
		tpr = dict()
		thresholds = dict()
		roc_auc = dict()
		# For test
		for i in range(self.NUM_CLASSES):
			 fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y[:, i], probs_list[:, i])
			 roc_auc[i] = metrics.auc(fpr[i], tpr[i])
		# Compute micro-average ROC curve and ROC area
		fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), probs_list.ravel())
		roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
		# Compute macro-average ROC curve and ROC area
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.NUM_CLASSES)]))
		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(self.NUM_CLASSES):
			 mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		# Finally average it and compute AUC
		mean_tpr /= self.NUM_CLASSES
		fpr["macro"] = all_fpr
		tpr["macro"] = mean_tpr
		roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
		return fpr, tpr, thresholds, roc_auc
    
	def randomColorGenerator(self, number_of_colors=1):
		return ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(number_of_colors)]
    
	def plot_roc_auc_curves(self, fpr, tpr, roc_auc, xlim=(-0.0025, 0.03), ylim=(0.99, 1.001), save_title=None):
		'''Plot ROC AUC Curves'''
		fig, axes = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(10,5))
		# Define
		lw = 2.5
		axes[0].set_xlabel('False Positive Rate')
		axes[1].set_xlabel('False Positive Rate')
		axes[0].set_ylabel('True Positive Rate')
		# Loop
		for i in range(self.NUM_CLASSES):
			axes[0].plot(fpr[i], tpr[i], color=self.class_colors[i], label='{0} ({1:0.2f})' ''.format(self.num_to_class[i], roc_auc[i]*100))
			axes[1].plot(fpr[i], tpr[i], color=self.class_colors[i], lw=lw, label='{0} ({1:0.2f})' ''.format(self.num_to_class[i], roc_auc[i]*100))
		# Left plot
		axes[0].plot(fpr['micro'], tpr['micro'], label='Micro avg ({:0.2f}%)' ''.format(roc_auc['micro']*100), linestyle=':', color='deeppink')
		axes[0].plot(fpr['macro'], tpr['macro'], label='Macro avg ({:0.2f}%)' ''.format(roc_auc['macro']*100), linestyle=':', color='navy')
		axes[0].plot([0, 1], [0, 1], color='k', linestyle='--')
		axes[0].scatter(0,1, label='Ideal', s=2)
		# Right plot
		axes[1].plot(fpr['micro'], tpr['micro'], lw=lw, label='Micro avg ({:0.2f}%)'.format(roc_auc['micro']*100), linestyle=':', color='deeppink')
		axes[1].plot(fpr['macro'], tpr['macro'], lw=lw, label='Macro avg ({:0.2f}%)'.format(roc_auc['macro']*100), linestyle=':', color='navy')
		axes[1].plot([0, 1], [0, 1], color='k', linestyle='--')
		axes[1].scatter(0,1, label='Ideal', s=50)
		# Limits
		axes[1].set_xlim(xlim)
		axes[1].set_ylim(ylim)
		# Grids
		axes[0].grid(True, linestyle='dotted', alpha=1)
		axes[1].grid(True, linestyle='dotted', alpha=1)
		# Legends
		axes[0].legend(loc=4)
		axes[1].legend(loc=4)
		# Plot
		plt.legend(loc="lower right")
		if save_title is not None:
			fig.tight_layout() 
			fig.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)
		plt.show()
        
	def process_data(self, folder, y_class, return_compressed=True):
		'''Get image data from folder'''
		imgs = []
		for i, filename in enumerate(sorted(os.listdir(folder))):        
			# We don't want .DS_Store or any other data files
			if filename.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
				# Load, Convert, Equalise, Standardise, Rescale
				img = cv2.imread(os.path.join(folder,filename))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = self.equalise_image(img, eq_type='HSV')
				if return_compressed:
					img = self.standardise_image(img)
					img = self.resize_image(img)
				# Append
				if img is not None:                
					imgs.append(img)
		# Concatenate
		if return_compressed:
			x_test = np.concatenate(imgs, axis=0)
			x_test = x_test.reshape(x_test.shape[0]//28, 28, 28, 3)
			y_test = np.array([y_class]*len(x_test))
		# Remove empty
		if return_compressed:
			indices_to_remove = []
			for i, x_i in enumerate(x_test):
				if x_i.std() == 0:
					indices_to_remove.append(i)
			x_test = np.delete(x_test, indices_to_remove, axis=0)
			y_test = np.delete(y_test, indices_to_remove, axis=0)
			# One hot encode if necessary
			if y_test.ndim == 1:
				y_test = self.one_hot_encode(y_test)
			return x_test, y_test
		return np.array(imgs)
        
	def get_detected_faces_cv(self, image, image_copy, scaleFactor = 1.1, cascade_path='data/haarcascades/haarcascade_frontalface_alt2.xml'):
		'''Returns detected faces's region of interests and coordinates using OpenCV'''
		rois = []
		coordinates_list = []
		cascade = cv2.CascadeClassifier(cascade_path)
		faces_rect = cascade.detectMultiScale(image_copy, scaleFactor=scaleFactor, minNeighbors=5)
		for i,(x, y, w, h) in enumerate(faces_rect):
			coordinates_list.append((x, y, w, h))
			rois.append(image_copy[y:y+h, x:x+w])
		return rois, coordinates_list
    
	def get_detected_faces_mtcnn(self, image, image_copy, detector):
		'''Returns detected faces's region of interests and coordinates using MTCNN'''
		rois = []
		coordinates_list = []
		faces_rect = detector.detect_faces(image_copy)
		for face in faces_rect:
			x, y, w, h = face['box']
			coordinates_list.append((x, y, w, h))
			rois.append(image_copy[y:y+h, x:x+w])
		return rois, coordinates_list
	
	def tag_images(self, images, all_faces, preds_reco, preds_fer, preds_gender, y_hat_prob_fer,
				   coordinates_list, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=5):
		'''Tags faces according to predicted class on the given image'''
		c = 0                                   # Counter
		for i, img in enumerate(images):        # For each image
			for j in range(len(all_faces[i])):  # For each face in each image
				# Calculate variables
				imageWidth, imageHeight, _ = img.shape
				x, y, w, h = coordinates_list[i][j]
				myColor = tuple([x_in*255 for x_in in mc.to_rgb(class_colors[preds_reco[c]])])
				myCoordinates = coordinates_list[i][j][0], coordinates_list[i][j][1]
				myLabelText = num_to_class_fer[preds_fer[c]]+' '+str(y_hat_prob_fer[c])
				myFontScale = (imageWidth * imageHeight) / (1000 * 1000)
				myFaceCenter = (int(x+w/2), int(y+h/2))
				myRadius = int(h/2)
				# Draw on image - rectangle for M, circle for F
				if preds_gender[c] == 0:
					rect = cv2.rectangle(img, (x, y), (x+w, y+h), myColor, thickness)
				else:
					circle = cv2.circle(img, myFaceCenter, myRadius, myColor, thickness)
				cv2.putText(img, org=(myCoordinates), text=myLabelText, fontFace=fontFace, 
							fontScale=myFontScale,color=(0,255,0), thickness=thickness)
				# Increment counter
				c += 1

	def load_final_test_images(self, folder, detection_method='cv'):
		'''Load the additional test images.'''
		# Lists
		pics = []
		copies = []
		all_faces = []
		coordinates_list = []
		normalised_faces = []
		unnormalised_faces = []
		# Method type
		#if detection_method == 'mtcnn':
		#	detector = MTCNN(min_face_size=50)
		# Get images
		print('Getting images...')
		for filename in sorted(os.listdir(folder)):
			pic_of_interest = cv2.imread(os.path.join(folder, filename))
			pic_of_interest = self.convertToRGB(pic_of_interest)
			pics.append(pic_of_interest)
		# Get face coordinates
		print('Getting face coordinates...')
		for pic in pics:
			image_copy = pic.copy()
			copies.append(image_copy)
			# Type
			#if detection_method == 'cv':
			faces_batch_temp, coordinates_temp = self.get_detected_faces_cv(pic, image_copy)
			#else:
			#	faces_batch_temp, coordinates_temp = self.get_detected_faces_mtcnn(pic, image_copy, detector)
			# Find fake-faces
			indices_to_remove = []
			for i, face in enumerate(faces_batch_temp):
				if face.size == 0:
					indices_to_remove.append(i)
			# Remove
			faces_batch = np.delete(faces_batch_temp, indices_to_remove, axis=0)
			coordinates = np.delete(coordinates_temp, indices_to_remove, axis=0)
			# Append
			all_faces.append(faces_batch)
			coordinates_list.append(coordinates)
		# Convert faces
		print('Processing faces...')
		for face_batch in all_faces:
			for i, face in enumerate(face_batch):
				if face.size != 0:
					unnormalised_faces.append(face)
					face = self.equalise_image(face, eq_type='HSV')
					face = self.standardise_image(face)
					normalised_faces.append(self.resize_image(face))
		print('Done!')
		return pics, copies, all_faces, coordinates_list, normalised_faces, unnormalised_faces

	def plot_before_after_tag(self, images, copies, save_title=None):
		'''Plot the before and after tag images side by side'''
		for i, img in enumerate(images):
			plt.figure(dpi=150)
			plt.imshow(np.hstack((copies[i], img)))
			plt.axis('off')
			if save_title is not None:
				plt.tight_layout()
				plt.savefig(f'{save_title}.pdf', bbox_inches='tight', format='pdf', dpi=200)
			plt.show()

	def restore_model(self, graph, graph_dir, checkpoint_dir):
		'''Import graph and restore model variables'''
		with graph.as_default():
			# Load graph and restore tf variables
			saver = tf.train.import_meta_graph(graph_dir)
			latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
			sess = tf.Session(graph=graph)
			saver.restore(sess, latest_checkpoint)
			# Get relevant tensors
			tf_cnn_softmax = graph.get_tensor_by_name('CNN/Softmax:0')
			tf_placeholder = graph.get_tensor_by_name('Placeholder:0')
		return sess, tf_cnn_softmax, tf_placeholder

	def run_model(self, sess, tf_placeholder, tf_cnn_softmax, x_test):
		'''Run model'''
		probs = sess.run(tf_cnn_softmax, feed_dict={tf_placeholder: x_test})
		y_hat = np.argmax(probs, axis=1)
		return probs, y_hat

	def run_keras_model(self, model, x_test):
		'''Run Keras Model'''
		probs = model.predict(np.array(x_test))
		y_hat = np.argmax(probs, axis=1)
		return probs, y_hat

	def restore_keras_graph(self, graph_dir_keras_fer, checkpoint_dir_keras_fer):
		'''Check if Keras model's graph is stored'''
		# For the sake of consistency, we also store Keras model's as checkpoint so we can use it as TF
		if not os.path.isfile('Keras_logdir/model.ckpt.meta'):
			model_fer = keras.models.load_model(graph_dir_keras_fer)
			saver_fer = tf.train.Saver()
			sess_fer = keras.backend.get_session()
			save_path = saver_fer.save(sess_fer, checkpoint_dir_keras_fer)