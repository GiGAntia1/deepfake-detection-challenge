#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 18 22:26:26 2020

@author: dale
"""

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

import os
import glob
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, ConvLSTM2D,\
BatchNormalization, MaxPooling2D, Flatten
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
FRAME_COUNT = 15
FRAME_HEIGHT = 448
FRAME_WIDTH = 448
FRAME_CHANNELS = 1
BATCH_SIZE = 1
EPOCHS = 1
STEPS_PER_EPOCH = 2000
FACE_DETECT = False
N_NETS = 11

# Image Processing
imgGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2,\
rotation_range=10, horizontal_flip=True, featurewise_center=False,\
featurewise_std_normalization=False, rescale=False, height_shift_range=0.2,\
width_shift_range=0.2, brightness_range=[0.6,1.2])

# Define custom video generators
genTrain = VideoFrameGenerator(phase='training', face_detection=FACE_DETECT,\
batch_size=BATCH_SIZE, nb_frames=FRAME_COUNT, nb_channel=FRAME_CHANNELS,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), transformation=imgGen, split=None,\
glob_pattern='./videos/training/{classname}/*.mp4')

genTest = VideoFrameGenerator(phase='training', face_detection=FACE_DETECT,\
batch_size=BATCH_SIZE, nb_frames=FRAME_COUNT, nb_channel=FRAME_CHANNELS,\
target_shape=(FRAME_HEIGHT,FRAME_WIDTH), transformation=None, split=None,\
glob_pattern='./videos/validation/{classname}/*.mp4')

# Calculate class weights for a balanced training data set
REAL = ['REAL' for i in glob.glob('./videos/training/REAL/*.mp4')]
FAKE = ['FAKE' for i in glob.glob('./videos/training/FAKE/*.mp4')]
y_train = REAL+FAKE
class_weights = class_weight.compute_class_weight('balanced',\
list(set(y_train)),y_train)
class_weights = dict(zip([0,1],class_weights))

# Calculate identify class labels for validation videos
REAL = [0 for i in glob.glob('./videos/validation/REAL/*.mp4')]
FAKE = [1 for i in glob.glob('./videos/validation/FAKE/*.mp4')]
y_val = REAL+FAKE

# Define custom metric
# Clip the ends of the predicted probabilities
def clip_loss(y_true, y_pred):
    clip_loss = binary_crossentropy(y_true, K.clip(y_pred, 0.05, 0.95))
    return clip_loss

# Define a new model
input_shape = Input((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS))
tower1 = ConvLSTM2D(3,(3,3), padding='same', activation='relu')(input_shape)
tower1 = BatchNormalization()(tower1)
tower1 = MaxPooling2D((2,2), strides=(1,1), padding='same')(tower1)
tower1 = Xception(include_top=False, weights='imagenet')(tower1)
tower1 = Flatten()(tower1)
out = Dropout(0.65)(tower1)
out = Dense(1, activation='sigmoid')(out)
model = Model(input_shape, out)

# Compile and view model summary
optimizer = Adam(learning_rate=1e-7)
model.compile(optimizer, loss='binary_crossentropy',\
metrics=['accuracy', clip_loss])
model.summary()

# Fit individual models and store predicted probabilities
model_preds = list()
for i in range(N_NETS):
    
    # Reset the states of all layers in the model
    model.reset_states()
    
    model.fit(genTrain, use_multiprocessing=True,\
    epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, workers=15,\
    max_queue_size=15, shuffle=True, class_weight=class_weights)

    # Model predicted probabilities
    yhati = model.predict(genTest, workers=15,\
    max_queue_size=15, use_multiprocessing=True, verbose=1)
    
    # Clip and store probabilities in the dictionary
    yhati = [np.clip(i, 0.05, 0.95) for i in yhati.flatten()]
    model_preds.append(yhati)
    
    # Print training round validation accuracy
    print('Round %.0f Complete' % (i+1))

# Calculate the cumulative performance boost
# with each added weak neural network
bootstrap = list()
for j in range(N_NETS):

    # Compute bootstrapped accuracy
    x = model_preds[:j+1]
    avgs = np.min(x, axis=0)
    bootstrap.append(log_loss(y_val, avgs))

# Visualize the log loss with each weak learner
plt.plot(bootstrap)
plt.xticks(range(N_NETS))


