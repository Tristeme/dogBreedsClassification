import tensorflow as tf
import os
import cv2
import numpy as np

# modelFile = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.h5'
modelFile = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.keras'

model = tf.keras.models.load_model(modelFile)

# print(model.summary())

inputShape = (311,311)
allLables = np.load("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsLabels.npy")
categories = np.unique(allLables)

import scipy.io


# prepare image
dir = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/Images/'
# Define the list of substrings
from sklearn.preprocessing import LabelEncoder

def prepareImage(img):
    resized = cv2.resize(img, inputShape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResul = imgResult / 255.
    return imgResul

imagePaths = [
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Pomeranian.jpg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Chihuahua.jpg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Afghan_hound.jpeg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/basset.webp',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Maltese_dog.jpeg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/papillon.JPG',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Pekinese.jpeg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/Shih-Tzu.jpeg',
    '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/toy_terrier.jpeg'
]

# # load image
# img = cv2.imread(testImagePath)
# imageForModel = prepareImage(img)
#
# # prediction
# resultArray = model.predict(imageForModel)
# answers = np.argmax(resultArray,axis=1)
# print(answers)
#
# text = categories[answers[0]]
# print(categories)
# print(text)
#
# font = cv2.FONT_HERSHEY_COMPLEX
# cv2.putText(img,text,(0,20),font,1,(209,19,77),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Initialize OpenCV window
# Display each image with its predicted breed

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for ax, path in zip(axes.flatten(), imagePaths):
    img = cv2.imread(path)
    if img is not None:
        imageForModel = prepareImage(img)
        resultArray = model.predict(imageForModel)
        predictedIndex = np.argmax(resultArray, axis=1)
        breedName = categories[predictedIndex[0]]

        # Convert image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img_rgb)
        ax.set_title(breedName)
        ax.axis('off')

plt.tight_layout()
plt.show()