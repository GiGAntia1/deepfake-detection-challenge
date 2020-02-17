#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script the defines the neural network architecture
using keras.

GPU WORKING CONFIGURATION:
Ubuntu 19.10
tensorflow-gpu 2.1
Cuda 10.1
libcudnn7 (7.6)
nvidia-driver-440
RTX 2070 GPU

@author: Dale Kube
"""

import os
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten, Conv2D, Concatenate, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import time

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

# Frame dimensions (per frame in each video)
FRAME_COUNT = 10
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
FRAME_CHANNELS = 3
shape = (FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)

input_shape = Input(shape=shape)
tower1 = ConvLSTM2D(16,(5,5), padding='same', activation='relu')(input_shape)
tower1 = Conv2D(3,(2,2), padding='same', activation='relu')(tower1)
tower1 = BatchNormalization()(tower1)
tower1 = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower1)
tower1 = Flatten()(tower1)
tower1 = Dense(500)(tower1)
tower1 = BatchNormalization()(tower1)

tower2 = ConvLSTM2D(3,(5,5), padding='same', activation='relu')(input_shape)
tower2 = ResNet50(include_top=False, weights='imagenet')(tower2)
tower2 = Flatten()(tower2)
tower2 = Dense(500)(tower2)
tower2 = BatchNormalization()(tower2)

merged = Concatenate(axis=-1)([tower1, tower2])
out = Dropout(0.25)(merged)
out = Dense(1, activation='sigmoid')(out)

model = Model(input_shape, out)

# Compile and view model summary
optimizer = SGD(learning_rate=1e-2, decay=1e-4)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model checkpoint
filepath = './checkpoints/weights-lstm-cnn-faces.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,\
save_best_only=True, save_weights_only=False, mode='min', period=1)

# Early Stopping
early_stop = EarlyStopping(monitor='loss', patience=3, mode='min')

# Combine callbacks
callbacks = [checkpoint, early_stop]

# Use the custom VideoFrameGenerator
datagen = VideoFrameGenerator(phase='training', face_detection=True, split=0.2,\
batch_size=1, nb_frames=FRAME_COUNT, target_shape=(FRAME_HEIGHT,FRAME_WIDTH),\
transformation=None, glob_pattern='./videos/{classname}/*.mp4')

# Fit model w/ data generators
start_time = time.time()

model.fit(datagen, use_multiprocessing=True, epochs=500, steps_per_epoch=3000,\
workers=14, max_queue_size=14, callbacks=callbacks, shuffle=True)

print("---model training finished in %s seconds ---" % (time.time()-start_time))
