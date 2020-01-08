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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Load the custom VideoFrameGenerator
parent_dir = '/mnt/extHDD/Kaggle/'
os.chdir(parent_dir)
from VideoFrameGenerator import VideoFrameGenerator

## WORKING CONFIGURATION (DO NOT CHANGE)
## Ubuntu 19.10
## tf-nightly-gpu 2.1
## Cuda 10.2
## libcudnn7 (7.6.5.32-1+cuda10.2)
## nvidia-driver-440
## RTX 2070 GPU

# TensorFlow version and GPU test
# Currently works with RTX 2070 card using 'tensorflow-gpu==2.0.0.beta1'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())

# Allow memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.50
session = InteractiveSession(config=config)

# Define model architecture
model = Sequential()

model.add(ConvLSTM2D(32,(3,3),input_shape=(10,500,500,3),padding='same',activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())

# LSTM Input size must be (timesteps, features)
#model.add(LSTM(25,input_shape=(10,750000)))
#model.add(Dropout(0.25))

#model.add(Dense(300))
#model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile and view model summary
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
# Save the model after each epoch if the validation loss decreases
# Stop the training if the validation loss does not improve
filepath = parent_dir+'checkpoints/weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,\
save_best_only=True, save_weights_only=False, mode='min', period=1)
early_stop = EarlyStopping(monitor='val_loss',patience=5,mode='min')
callbacks = [checkpoint,early_stop]

# Define the primary VideoFrameGenerator
img_gen = ImageDataGenerator(shear_range=0.1, horizontal_flip=True)
datagen = VideoFrameGenerator(classes=['FAKE','REAL'], split=0.2,\
target_shape=(500,500), transformation=None, batch_size=1, nb_frames=20,
use_frame_cache=True, shuffle=True)

# Fit model w/ data generators
model.fit(datagen, use_multiprocessing=True, epochs=10, steps_per_epoch=100,\
workers=3, callbacks=callbacks, shuffle=True)


