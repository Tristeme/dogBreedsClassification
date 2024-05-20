import numpy as np
import os
import pandas as pd

IMAGE_SIZE = (311, 311)
IMAGE_FULL_SIZE = (311, 311, 3)

trainMyImageFolder = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/train'

# Setting the path to the folder where the pictures are located
dir = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/Images/'

pomeranian_dir = dir+'n02112018-Pomeranian'
chihuahua_dir = dir+'n02085620-Chihuahua'
japanese_spaniel_dir = dir+'n02085782-Japanese_spaniel'
maltese_dir = dir+'n02085936-Maltese_dog'
pekinese_dir = dir+'n02086079-Pekinese'
shitzu_dir = dir+'n02086240-Shih-Tzu'
blenheim_spaniel_dir = dir+'n02086646-Blenheim_spaniel'
papillon_dir = dir+'n02086910-papillon'
toy_terrier_dir = dir+'n02087046-toy_terrier'
afghan_hound_dir = dir+'n02088094-Afghan_hound'
basset_dir = dir+'n02088238-basset'


import xml.etree.ElementTree as ET

# Function to read bounding box from annotation file
def read_annotation(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return (xmin, ymin, xmax, ymax)


import cv2 # import Open CV
allImages = []
allLables = []
imgsize = 150
# Create a list to store image names and corresponding species
data = []
# Define a function to read in dog pictures
def training_data(label, data_dir):
    print ("reading in：", data_dir)
    for img in os.listdir(data_dir):
        path = os.path.join(data_dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        annotation_dir = path.replace('.jpg', '').replace('Images', 'Annotation')
        # Read the annotation for the current image
        xmin, ymin, xmax, ymax = read_annotation(annotation_dir)
        cropped_img = img[ymin:ymax, xmin:xmax]  # Crop the region of interest
        resized = cv2.resize(cropped_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        data.append({'id': path, 'breed': label})

        allImages.append(resized)
        allLables.append(str(label))

        for img in os.listdir(data_dir):
            if data_dir.endswith('.jpg'):
                # Add the file name and corresponding species to the list
                data.append({'id': img, 'breed': label})
# Read dog pictures from 9 catalogues
training_data('pomeranian',pomeranian_dir)
training_data('chihuahua',chihuahua_dir)
training_data('maltese',maltese_dir)
training_data('pekinese',pekinese_dir)
training_data('shitzu',shitzu_dir)
training_data('papillon',papillon_dir)
training_data('toy_terrier',toy_terrier_dir)
training_data('afghan_hound',afghan_hound_dir)
training_data('basset',basset_dir)

# Creating a DataFrame using a list
df = pd.DataFrame(data)

# Saving a DataFrame as a CSV file
df.to_csv('/csse/users/yzo17/PycharmProjects/dogBreedsClassification/labels.csv', index=False)

# load the csv file
df = pd.read_csv('/csse/users/yzo17/PycharmProjects/dogBreedsClassification/labels.csv')

grouplabels = df.groupby('breed')["id"].count()

# prepare all the images and labels as Numpy arrays

print(len(allImages))
print(len(allLables))
print('save the data:')
np.save("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsImages.npy", allImages)
np.save("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsLabels.npy", allLables)
print('finish the data')