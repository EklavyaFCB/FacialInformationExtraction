import pandas as pd
import numpy as np
import argparse
import cv2
import sys
import os

def convert_images_to_csv(args, img_size=28):
  '''Takes in directory path and returns the compressed images's pixels as a list'''

  images = []

  # Iterate through images in folder
  for i, filename in enumerate(sorted(os.listdir(args.input_dir))):
      if filename.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:

        # Print progress
        if args.debug:
          percent = int((i+1)/len(os.listdir(args.input_dir))*100)
          if percent%10==0:
            print(percent,'%')
        
        # Load and append
        img = cv2.imread(os.path.join(args.input_dir, filename))
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
  
  # Convert
  images = np.array(images)

  if args.debug:
    print('Saving as .csv file ...')

  # Save images as .csv
  np.savetxt(args.output_dir+'/'+args.filename+'.csv', 
             images.reshape(images.shape[0], np.prod(images.shape[1:])), 
             delimiter=",",
             fmt='%i')

def main():
  parser = argparse.ArgumentParser(description='Convert images in a folder to .csv')
  parser.add_argument('-d','--debug', action='store_true', default=False, help='Print progress to stderr')
  parser.add_argument('-i','--input_dir', type=str, action='store', help='Give input directory path')
  parser.add_argument('-o','--output_dir', type=str, action='store', help="Give output directory path")
  parser.add_argument('-f','--filename', type=str, default='image_data', action='store', help='Give output file name')
  args = parser.parse_args()

  images = convert_images_to_csv(args)

if __name__ == "__main__":
  main()