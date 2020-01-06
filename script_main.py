#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script the defines the neural network architecture
using keras.

Article References:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://pypi.org/project/keras-video-generators/

@author: Dale Kube
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Load the custom VideoFrameGenerator
parent_dir = '/mnt/extHDD/Kaggle/'
os.chdir(parent_dir)
from VideoFrameGenerator import VideoFrameGenerator

# TensorFlow version and GPU test
# Currently works with RTX 2070 card using 'tensorflow-gpu==2.0.0.beta1'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
#print(tf.config.list_physical_devices('GPU'))
#tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)

# Allow memory growth
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define model architecture
model = Sequential()

# Input size must be (timesteps, features)
model.add(ConvLSTM2D(25,(3,3),input_shape=(10,500,500,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy',
              metrics=['accuracy'])

# View model summary
model.summary()

# Define callbacks
# Save the model after each epoch if the validation loss decreases
filepath = parent_dir+'checkpoints/weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,\
save_best_only=True, save_weights_only=False, mode='min', period=1)
early_stop = EarlyStopping(monitor='val_loss',patience=5,mode='min')
callbacks = [checkpoint,early_stop]

# Define the primary VideoFrameGenerator
img_gen = ImageDataGenerator(shear_range=0.1, horizontal_flip=True)
datagen = VideoFrameGenerator(classes=['FAKE','REAL'], split=0.2,\
target_shape=(500,500), transformation=None, batch_size=16)

# Fit model w/ data generators
model.fit(datagen, use_multiprocessing=True, epochs=10, steps_per_epoch=100,\
workers=10, callbacks=callbacks)
