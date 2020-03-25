#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script the defines the neural network architecture
using keras.

Article References:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://pypi.org/project/keras-video-generators/

GPU WORKING CONFIGURATION:
Ubuntu 19.10
tensorflow-gpu 2.1
Cuda 10.1
libcudnn7 (7.6)
nvidia-driver-440
RTX 2070 GPU

@author: Dale Kube
"""

import os, glob, time
from sklearn.utils import class_weight
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten, GaussianNoise
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# TensorFlow version and GPU availability test
print(tf.__version__)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
assert len(gpu_devices) > 0, "GPU Not Found"
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.compat.v1.disable_eager_execution() 

# Allow memory growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load the custom VideoFrameGenerator
parent_dir = '/mnt/extHDD/Kaggle/'
os.chdir(parent_dir)
from VideoFrameGenerator import VideoFrameGenerator

# Key elements for training
FRAME_COUNT = 20
FRAME_WIDTH = 299
FRAME_HEIGHT = 299
FRAME_CHANNELS = 1
BATCH_SIZE = 2
EPOCHS = 250
STEPS_PER_EPOCH = 500
FACE_DETECT = False
FILEPATH = './checkpoints/weights-20200324.hdf5'

# Load existing model
# from tensorflow.keras.models import load_model
# model = load_model('./checkpoints/weights-20200320.hdf5')

# Define a new model
input_shape = Input((FRAME_COUNT, FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS))

LSTM = ConvLSTM2D(3,(3,3), strides=(1,1), padding='same', activation='relu',\
recurrent_activation='relu', dropout=0.75, recurrent_dropout=0.75)(input_shape)
LSTM = BatchNormalization()(LSTM)
LSTM = MaxPooling2D((2,2), strides=(1,1), padding='same')(LSTM)

XCEPT = Xception(include_top=False, weights='imagenet', pooling='max')(LSTM)
XCEPT = Flatten()(XCEPT)
XCEPT = GaussianNoise(0.25)(XCEPT)

OUT = Dropout(0.75)(XCEPT)
OUT = Dense(1, activation='sigmoid')(OUT)

model = Model(input_shape, OUT)

# Define custom metric
# Clip the ends of the predicted probabilities
def clip_loss(y_true, y_pred):
    clip_loss = binary_crossentropy(y_true, K.clip(y_pred, 0.01, 0.99))
    return clip_loss

# Compile and view model summary
optimizer = SGD(learning_rate=1e-6)
model.compile(optimizer, loss='binary_crossentropy',\
metrics=['accuracy', clip_loss])
model.summary()

# Model checkpoint
checkpoint = ModelCheckpoint(FILEPATH, monitor='val_clip_loss', verbose=1,\
save_best_only=True, save_weights_only=False, mode='min', period=1)

# Early Stopping
early_stop = EarlyStopping(monitor='loss', patience=5, mode='min')

# Combine callbacks
callbacks = [checkpoint, early_stop]

# Image Processing
imgGen = ImageDataGenerator(shear_range=0.1, zoom_range=0.3,\
rotation_range=30, horizontal_flip=True, featurewise_center=False,\
featurewise_std_normalization=False, rescale=False, height_shift_range=0.3,\
width_shift_range=0.3, brightness_range=[0.6,1.4])

# Define custom video generators
genTrain = VideoFrameGenerator(phase='training', face_detection=FACE_DETECT,\
split=None, batch_size=BATCH_SIZE, nb_frames=FRAME_COUNT,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), transformation=imgGen,\
glob_pattern='./videos/training/{classname}/*.mp4', nb_channel=FRAME_CHANNELS)

genVal = VideoFrameGenerator(phase='training', face_detection=FACE_DETECT,\
split=None, batch_size=BATCH_SIZE, nb_frames=FRAME_COUNT,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), transformation=imgGen,\
glob_pattern='./videos/validation/{classname}/*.mp4', nb_channel=FRAME_CHANNELS)

# Calculate class weights for a balanced training data set
REAL = ['REAL' for i in glob.glob('./videos/training/REAL/*.mp4')]
FAKE = ['FAKE' for i in glob.glob('./videos/training/FAKE/*.mp4')]
y_train = REAL+FAKE
class_weights = class_weight.compute_class_weight('balanced',\
list(set(y_train)),y_train)
class_weights = dict(zip([0,1],class_weights))

# Fit model
start_time = time.time()

model.fit(genTrain, validation_data=genVal, use_multiprocessing=True,\
epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, workers=15,\
max_queue_size=30, callbacks=callbacks, shuffle=True,\
class_weight=class_weights)

# Print completion time
finish_time = (time.time()-start_time)/3600
print("---model training finished in %.2f hours---" % (finish_time))

# Define the VideoFrameGenerator, specific to frame dimensions
testgen = VideoFrameGenerator(phase='predict', face_detection=FACE_DETECT,\
split=None, batch_size=BATCH_SIZE, nb_frames=FRAME_COUNT,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), nb_channel=FRAME_CHANNELS,\
glob_pattern='./test_videos/*.mp4', transformation=None)

# Apply the generator and make predictions with the model
probs = model.predict(testgen, verbose=1, workers=15,\
use_multiprocessing=True, max_queue_size=30)

# Plot the predicted probabilities
plt.hist(probs, bins=20)
