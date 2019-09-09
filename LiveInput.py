import cv2
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential
from myutilitymethods import MyMethods
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

# ========== DEFINE VARIABLES ==========
NUM_CLASSES = 4
NUM_CLASSES_FER = 7

class_colors = ['C1', 'C2', 'C3', 'C0']
bgr_colors = [(0,165,255), (0,255,0), (0,0,255), (255,0,0)]

num_to_class_gender = ['M', 'F']
num_to_class = ['Myself', 'Sister', 'Mother', 'Father']
num_to_class_fer = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class_to_num = {'Myself': 0, 'Sister': 1, 'Mother': 2, 'Father': 3,}
class_to_num_fer = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral':4, 'sad':5, 'surprise':6}

path = 'data/haarcascades/haarcascade_frontalface_alt2.xml'
mm = MyMethods(NUM_CLASSES, num_to_class, class_to_num)
mm_fer = MyMethods(NUM_CLASSES_FER, num_to_class_fer, class_to_num_fer)
fontFace=cv2.FONT_HERSHEY_SIMPLEX
thickness = 5
green_color = (0,255,0)

# ========== METHODS ==========
def recreate_keras_model(num_features, nb_classes, edge, channels):
	'''Re-creates Keras model'''
	model = keras.Sequential()
	
	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(edge, edge, channels), kernel_regularizer=l2(0.01)))
	model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	
	model.add(Dense(2*2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2*2*num_features, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(2*num_features, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(nb_classes, activation='softmax'))

	return model

# ========== MAIN ==========
def main():

	# Define Argsparser
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-a', '--allModels', action='store_true', default=True, help='Run all three models: face recognition, gender classification, and emotion detection.')
	parser.add_argument('-fr', '--faceReco', action='store_true', default=False, help='Run only face recognition model.')
	parser.add_argument('-gc', '--genderClass', action='store_true', default=False, help='Run only gender classification model.')
	parser.add_argument('-ed', '--emotionDetect', action='store_true', default=False, help='Run only emotion detection model.')
	args = parser.parse_args()

	# XOR only
	if args.faceReco or args.emotionDetect or args.genderClass:
		args.allModels = False
	
	# Open OpenCV Camera
	cap = cv2.VideoCapture(0)
	print(cap.isOpened())
	
	# Restore graphs
	graph_reco = tf.Graph() 
	graph_fer = tf.Graph()
	graph_gender = tf.Graph()

	#detector = MTCNN(min_face_size=50)
	
	# Restore reco model
	if args.allModels or args.faceReco:
		graph_dir_reco = 'Home_logdir/-149.meta'
		checkpoint_dir_reco = 'Home_logdir/'
		reco_values = mm.restore_model(graph_reco, graph_dir_reco, checkpoint_dir_reco)
		sess_reco = reco_values[0]
		tf_cnn_softmax_reco = reco_values[1]
		tf_placeholder_reco = reco_values[2]
	
	# Restore Keras FER model
	if args.allModels or args.emotionDetect:
		weights_dir_keras_fer = 'Keras_logdir/my_keras_model_weights.h5'
		graph_dir_keras_fer = 'Keras_logdir/my_keras_model.h5'
		checkpoint_dir_keras_fer = 'Keras_logdir/model.ckpt'
		#model_fer = keras.models.load_model(graph_dir_keras_fer)
		model_fer = recreate_keras_model(64, NUM_CLASSES_FER, 28, 3)
		model_fer.load_weights(weights_dir_keras_fer)
	
	# Restore Gender model
	if args.allModels or args.genderClass:
		graph_dir_gender = 'Gender_logdir/-74.meta'
		checkpoint_dir_gender = 'Gender_logdir/'
		gender_values = mm.restore_model(graph_gender, graph_dir_gender, checkpoint_dir_gender)
		sess_gender = gender_values[0]
		tf_cnn_softmax_gender = gender_values[1]
		tf_placeholder_gender = gender_values[2]
	
	frames = []

	# Run camera
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frames.append(rgb_frame)

		# Copy
		image_copy = rgb_frame.copy()

		# Get heads
		cascade = cv2.CascadeClassifier(path)
		faces_rect = cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)
		#faces_rect = detector.detect_faces(rgb_frame)

		# Plot rectangle(s) for each face in each frame
		for face in faces_rect:
			x, y, w, h = face
			#x, y, w, h = face['box']
			rois = [image_copy[y:y+h, x:x+w]]

			if face.size != 0:
				norm_face = mm.equalise_image(rois[0], eq_type='HSV')
				norm_face = mm.standardise_image(norm_face)
				norm_face = mm.resize_image(norm_face)
			
			if args.allModels or args.faceReco:
				probs_reco, y_hat_reco = mm.run_model(sess_reco, tf_placeholder_reco,   tf_cnn_softmax_reco, norm_face[None,:,:,:])
			
			if args.allModels or args.emotionDetect:
				probs_fer, y_hat_fer = mm.run_keras_model(model_fer, norm_face[None,:,:,:])

			if args.allModels or args.genderClass:
				probs_gender, y_hat_gender = mm.run_model(sess_gender, tf_placeholder_gender,  tf_cnn_softmax_gender, norm_face[None,:,:,:])

			# Define variables
			#myLabelText = num_to_class_gender[y_hat_gender[0]]+'-'+num_to_class_fer[y_hat_fer[0]]
			myLabelText = num_to_class_fer[y_hat_fer[0]]+' '+str(np.round(np.max(probs_fer, axis=1), 2)[0])
			myColor = bgr_colors[y_hat_reco[0]]
			myFontScale = (frame.shape[0] * frame.shape[1]) / (1000 * 1000)
			myFaceCenter = (int(x+w/2), int(y+h/2))
			myRadius = int(h/2)
			myCoordinates = x, y

			if y_hat_gender[0] == 0:
				rect = cv2.rectangle(frame, (x, y), (x+w, y+h), myColor, thickness=2)
			else:
				circle = cv2.circle(frame, myFaceCenter, myRadius, myColor, thickness=2)

			# Tag
			cv2.putText(frame, 
							org=(myCoordinates), 
							text=myLabelText, 
							fontFace=fontFace, 
							fontScale=myFontScale, 
							color=green_color)

		# Display the resulting frame
		cv2.imshow('frame', frame)
	
		# Release keys
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()