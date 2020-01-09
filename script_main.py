#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script the defines the neural network architecture
using keras.

Article References:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://pypi.org/project/keras-video-generators/

CuDNN Issue References:
https://github.com/tensorflow/tensorflow/issues/34695
https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error

@author: Dale Kube
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Load the custom VideoFrameGenerator
parent_dir = '/mnt/extHDD/Kaggle/'
os.chdir(parent_dir)
from VideoFrameGenerator import VideoFrameGenerator

## CPU WORKING CONFIGURATION (DO NOT CHANGE)
## Ubuntu 19.10
## tf-nightly-gpu 2.1
## Cuda 10.2
## libcudnn7 (7.6.5.32-1+cuda10.2)
## nvidia-driver-440
## RTX 2070 GPU

## GPU WORKING CONFIGURATION (DO NOT CHANGE)
## Ubuntu 19.10
## tensorflow-gpu 2.1
## Cuda 10.1
## libcudnn7 (7.6)
## nvidia-driver-440
## RTX 2070 GPU

# TensorFlow version and GPU availability test
print(tf.__version__)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
assert len(gpu_devices) > 0, "GPU Not Found"
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# Allow memory growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.50
session = InteractiveSession(config=config)

# Frame dimensions (per frame in each video)
FRAME_COUNT = 15
FRAME_HEIGHT = 256
FRAME_WIDTH = 256
FRAME_CHANNELS = 3
shape = (FRAME_COUNT,FRAME_HEIGHT,FRAME_WIDTH,FRAME_CHANNELS)

# Define sequential model architecture
model = Sequential()

model.add(ConvLSTM2D(32,(5,5),input_shape=shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())
model.add(Flatten())

# LSTM Input size must be (timesteps, features)
#model.add(LSTM(25,input_shape=(10,750000)))
#model.add(Dropout(0.25))

model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile and view model summary
optimizer = SGD(learning_rate=0.001)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
# Save the model after each epoch if the validation loss decreases
# Stop the training if the validation loss does not improve
filepath = parent_dir+'checkpoints/weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,\
save_best_only=True, save_weights_only=False, mode='min', period=1)
early_stop = EarlyStopping(monitor='loss',patience=10,mode='min')
callbacks = [checkpoint,early_stop]

# Use the custom VideoFrameGenerator
img_gen = ImageDataGenerator(shear_range=0.1, horizontal_flip=True)

datagen = VideoFrameGenerator(split=0.2, batch_size=1, nb_frames=15,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), transformation=img_gen)

# Fit model w/ data generators
model.fit(datagen, use_multiprocessing=True, epochs=1000, steps_per_epoch=100,\
workers=10, max_queue_size=10, callbacks=callbacks, shuffle=True)


