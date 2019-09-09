import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2
import os

def convertToRGB(image):
	'''Convert image from BGR to RGB'''
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_detected_faces(cascade, image, scaleFactor = 1.1, output_dir = '/Users/Eklavya/Movies/SortedFaces/Dad/', date_format = "%Y_%m_%d_%H_%M_%S_%f"):
	'''Saves the cropped faces in the given output directory'''
	rois = []
	image = convertToRGB(image)
	# Apply the haar classifier to detect faces
	faces_rect = cascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=5)
	# For each face, get the region of interest and save it
	for i,(x, y, w, h) in enumerate(faces_rect):
		roi = image[y:y+h, x:x+w]
		cv2.imwrite(f'{output_dir}/{datetime.datetime.now().strftime(date_format)[:-4]}.png', convertToRGB(roi))

def get_images(folder, cascade):
	'''Takes in directory path and returns the images's pixels as a list'''
	images = []
	for filename in sorted(os.listdir(folder)):
		# Load, convert
		img = cv2.imread(os.path.join(folder, filename))
		# Append
		if img is not None:
			save_detected_faces(cascade, img)

def main():
	haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
	images = get_images('/Users/Eklavya/Movies/DadGooglePhotos/', haar_cascade_face)

if __name__ == "__main__":
	main()