import numpy as np
import cv2
import tensorflow

IMAGE_SIZE = (311, 311)
IMAGE_FULL_SIZE = (311,311,3)
batchSize = 8

allImages = np.load("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsImages.npy")
allLables = np.load("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsLabels.npy")

# convert the lables text to integers
print('allLables', allLables)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
integerLables = le.fit_transform(allLables)
print('integerLables', integerLables)

# unique interger lables
numOfCategories = len(np.unique(integerLables))
print(numOfCategories)

# convert the interger lables to categorical -> prepare for the train
from tensorflow.keras.utils import to_categorical

allLablesForModel = to_categorical(integerLables, num_classes = numOfCategories)
# print(allLablesForModel)

# normalize the images from 0-255 to 0-1
allImagesForModel = allImages / 255.0

# create train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    allImagesForModel, allLablesForModel, test_size=0.3, random_state=42
)


print("X_train,X_test,y_train shape:")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# free some memory
del allImages
del allLables
del integerLables
del allImagesForModel

# build the model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge

# nitialize the NASNetLarge model with input_shape=IMAGE_FULL_SIZE,
# indicating the input image size and weights='imagenet', which loads pre-trained weights from the ImageNet dataset.
myModel = NASNetLarge(input_shape=IMAGE_FULL_SIZE, weights='imagenet', include_top=False)

# we dont want to train the existing layers
# 1. Freezing Layers: After initializing the NASNetLarge model,
# you iterate through its layers and set layer.trainable = False for each layer.
# This freezes the weights of the pre-trained layers, preventing them from being updated during training.
for layer in myModel.layers:
    layer.trainable = False
    print(layer.name)

# add Flatten layer
plusFlattenLayer = Flatten()(myModel.output)

# add the last dense layer without 3 classes
prediction = Dense(numOfCategories, activation='softmax')(plusFlattenLayer)

# 2. Adding New Layers: You add new layers on top of the pre-trained NASNetLarge model.
# Specifically, you add a Flatten layer and a Dense layer with numOfCategories units and softmax activation,
# which is suitable for multi-class classification.
model = Model(inputs=myModel.input, outputs=prediction)

# print(model.summary())

from tensorflow.keras.optimizers import Adam

learning_rate = 1e-4 # 0.0001
opt = Adam(learning_rate)
# Compiling the Model:
# compiled the model with an optimizer (Adam), a loss function (categorical_crossentropy), and a metric (accuracy).
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
stepsPerEpoch = np.ceil(len(X_train) / batchSize)
validationSteps = np.ceil(len(X_test) / batchSize)

# early stopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# best_model_file = "/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.h5"
best_model_file = "/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.keras"

callbacks_list = [ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
                  ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1,mode='auto', min_lr=1e-6),
                  EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)]

# callbacks_list = [ModelCheckpoint(best_model_file, verbose=1, save_best_only=True)]



from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Setting Data Enhancement Parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# usedatagen.flow() to treat X_train and y_train
train_generator = datagen.flow(
    X_train,
    y_train,
    batch_size=batchSize
)
# Fetch a batch of images
x_batch, y_batch = next(train_generator)

# x_batch contains the images, and y_batch contains the corresponding labels
# Since the batch size is 1, we can directly access the first image
img = x_batch[0]

# Expand the dimensions of img from (311, 311, 3) to (1, 311, 311, 3)
img_expanded = np.expand_dims(img, axis=0)  # Now img_expanded has shape (1, 311, 311, 3)

r = model.fit(
    train_generator,
    #steps_per_epoch=int(stepsPerEpoch),
    validation_data=(X_test, y_test),
    batch_size=batchSize,
    epochs=30,
    callbacks=callbacks_list,
)
print(r,111)

# # Evaluate the model on test data
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)
print(r.history)
# Plot training & validation accuracy values
# Create subplots
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(r.history['accuracy'], label='Train Accuracy')
plt.plot(r.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(r.history['loss'], label='Train Loss')
plt.plot(r.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

# Show the plots
plt.show()
