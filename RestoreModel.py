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
import albumentations as A
from IPython import display
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import matplotlib.colors as mc
from myutilitymethods import MyMethods
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix as sk_cm
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint


# ========== METHODS ==========
def process_data(folder, y_class, return_compressed=True):
    '''Get image data from folder'''
    imgs = []
    
    for i, filename in enumerate(sorted(os.listdir(folder))):
            
        # We don't want .DS_Store or any other data files
        if filename.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
            
            # Load, Convert, Equalise, Standardise, Rescale
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = mm.equalise_image(img, eq_type='HSV')

            if return_compressed:
                img = mm.standardise_image(img)
                img = mm.resize_image(img)
            
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
    
        if y_test.ndim == 1:
            y_test = mm.one_hot_encode(y_test)
                
        return x_test, y_test
    
    return np.array(imgs)

def get_detected_faces_cv(image, image_copy, scaleFactor = 1.1, cascade_path='data/haarcascades/haarcascade_frontalface_alt2.xml'):
    '''Returns detected faces's region of interests and coordinates using OpenCV'''
    rois = []
    coordinates_list = []
    cascade = cv2.CascadeClassifier(cascade_path)
    faces_rect = cascade.detectMultiScale(image_copy, scaleFactor=scaleFactor, minNeighbors=5)
    for i,(x, y, w, h) in enumerate(faces_rect):
        coordinates_list.append((x, y, w, h))
        rois.append(image_copy[y:y+h, x:x+w])
    return rois, coordinates_list

def get_detected_faces_mtcnn(image, image_copy, detector):
    '''Returns detected faces's region of interests and coordinates using MTCNN'''
    rois = []
    coordinates_list = []
    faces_rect = detector.detect_faces(image_copy)
    for face in faces_rect:
        x, y, w, h = face['box']
        coordinates_list.append((x, y, w, h))
        rois.append(image_copy[y:y+h, x:x+w])
    return rois, coordinates_list

def tag_images(images, all_faces, preds_reco, preds_fer, preds_gender, y_hat_prob_fer, coordinates_list, fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=5):
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

def load_final_test_images(folder='TestImages/', detection_method='cv'):
    
    pics = []
    copies = []
    all_faces = []
    coordinates_list = []
    normalised_faces = []
    unnormalised_faces = []
    
    if detection_method == 'mtcnn':
        detector = MTCNN(min_face_size=50)
    
    # Get images
    print('Getting images...')
    for filename in sorted(os.listdir(folder)):
        pic_of_interest = cv2.imread(os.path.join(folder, filename))
        pic_of_interest = mm.convertToRGB(pic_of_interest)
        pics.append(pic_of_interest)

    # Get face coordinates
    print('Getting face coordinates...')
    for pic in pics:
        image_copy = pic.copy()
        copies.append(image_copy)
        
        if detection_method == 'cv':
            faces_batch_temp, coordinates_temp = get_detected_faces_cv(pic, image_copy)
        else:
            faces_batch_temp, coordinates_temp = get_detected_faces_mtcnn(pic, image_copy, detector)
        
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
                face = mm.equalise_image(face, eq_type='HSV')
                face = mm.standardise_image(face)
                normalised_faces.append(mm.resize_image(face))
            
    print('Done!')
    return pics, copies, all_faces, coordinates_list, normalised_faces, unnormalised_faces

def plot_before_after_tag(images, copies):

    for i, img in enumerate(images):
        plt.figure(dpi=150)
        plt.imshow(np.hstack((copies[i], img)))
        plt.axis('off')
        plt.tight_layout()
        plt.legend(lines, ['Myself', 'Sister', 'Mother', 'Father'], 
                   bbox_to_anchor=(0.0, 1.15), loc="upper left", ncol=4)
        #rand_name = str(np.random.randint(1, 100000))
        #plt.savefig(f'{rand_name}.pdf', bbox_inches='tight', format='pdf', dpi=200)
        plt.show()

def restore_model(graph, graph_dir, checkpoint_dir):
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

def run_model(sess, tf_placeholder, tf_cnn_softmax, x_test):
    '''Run model'''
    probs = sess.run(tf_cnn_softmax, feed_dict={tf_placeholder: x_test})
    y_hat = np.argmax(probs, axis=1)
    return probs, y_hat

def run_keras_model(model, x_test):
    '''Run Keras Model'''
    probs = model.predict(np.array(x_test))
    y_hat = np.argmax(probs, axis=1)
    return probs, y_hat

def restore_keras_graph(graph_dir_keras_fer, checkpoint_dir_keras_fer):
    '''Check if Keras model's graph is stored'''
    # For the sake of consistency, we also store Keras model's 
    # as checkpoint so we can use it as TF
    if not os.path.isfile('Keras_logdir/model.ckpt.meta'):
        model_fer = keras.models.load_model(graph_dir_keras_fer)
        saver_fer = tf.train.Saver()
        sess_fer = keras.backend.get_session()
        save_path = saver_fer.save(sess_fer, checkpoint_dir_keras_fer)

# ========== DEFINE VARIABLES ==========
NUM_CLASSES = 4
class_colors = ['C1', 'C2', 'C3', 'C0']
num_to_class = ['Myself', 'Sister', 'Mother', 'Father']
class_to_num = {'Myself': 0, 'Sister': 1, 'Mother': 2, 'Father': 3,}

NUM_CLASSES_FER = 7
num_to_class_fer = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
class_to_num_fer = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral':4, 'sad':5, 'surprise':6}

mm = MyMethods(NUM_CLASSES, num_to_class, class_to_num)
mm_fer = MyMethods(NUM_CLASSES_FER, num_to_class_fer, class_to_num_fer)

lines = [Line2D([0], [0], color=c) for c in class_colors]

# ========== MAIN ==========
def main():

    # Create new graphs
    graph_reco = tf.Graph() 
    graph_fer = tf.Graph()
    graph_gender = tf.Graph()

    # Restore Reco model
    graph_dir_reco = 'Home_logdir/-149.meta'
    checkpoint_dir_reco = 'Home_logdir/'
    sess_reco, tf_cnn_softmax_reco, tf_placeholder_reco = restore_model(graph_reco, graph_dir_reco, checkpoint_dir_reco)

    # Restore Keras FER model
    weights_dir_keras_fer = 'Keras_logdir/my_keras_model_weights.h5'
    model_fer = recreate_keras_model(64, NUM_CLASSES_FER, 28, 3)
    model_fer.load_weights(weights_dir_keras_fer)

    # Restore Gender model
    graph_dir_gender = 'Gender_logdir/-74.meta'
    checkpoint_dir_gender = 'Gender_logdir/'
    sess_gender, tf_cnn_softmax_gender, tf_placeholder_gender = restore_model(graph_gender, graph_dir_gender, checkpoint_dir_gender)

    #Test
    face_values = load_final_test_images(detection_method='cv')

    pics = face_values[0]
    copies = face_values[1]
    all_faces = face_values[2]
    coordinates_list = face_values[3]
    normalised_faces = face_values[4]
    unnormalised_faces = face_values[5]

    # Run
    probs_reco, y_hat_reco = run_model(sess_reco, tf_placeholder_reco, tf_cnn_softmax_reco, normalised_faces)
    probs_fer, y_hat_fer = run_keras_model(model_fer, normalised_faces)
    probs_gender, y_hat_gender = run_model(sess_gender, tf_placeholder_gender, tf_cnn_softmax_gender, normalised_faces)
    y_hat_prob_fer = np.round(np.max(probs_fer, axis=1), 2)

    # Tag and plot
    tag_images(pics, all_faces, y_hat_reco, y_hat_fer, y_hat_gender, y_hat_prob_fer, coordinates_list)
    plot_before_after_tag(pics, copies)

if __name__ == "__main__":
    main()
