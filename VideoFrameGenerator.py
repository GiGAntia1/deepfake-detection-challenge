#!/usr/bin/env python

"""
VideoFrameGenerator w/ Face Detection
--------------------------------------
A simple frame generator that takes distributed frames from
videos. It is useful for videos that are scaled from frame 0 to end
and that have no noise frames.

https://github.com/metal3d/keras-video-generators/blob/master/src/keras_video/generator.py
https://github.com/keras-team/keras/issues/12586
https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
"""

import os
import glob
import numpy as np
import cv2
from math import floor
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import dlib
detector = dlib.get_frontal_face_detector()

class VideoFrameGenerator(Sequence):
    
    """
    Create a generator that returns batches of frames from video
    - phase: str, the phase of model deployment, either 'training' or 'predict'
    - face_detection: bool, should faces be cropped from the frames?
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and test
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - _validation_data: already filled list of data, **do not touch !**
    You may use the "classes" property to retrieve the class list afterward.
    The generator has that properties initialized:
    - classes_count: number of classes that the generator manages
    - files_count: number of video that the generator can provides
    - classes: the given class list
    - files: the full file list that the generator will use, this \
        is usefull if you want to remove some files that should not be \
        used by the generator.
    """

    def __init__(
            self,
            phase: str = 'training',
            face_detection: bool = False,
            rescale=1/255.,
            nb_frames: int = 10,
            classes: list = ['FAKE','REAL'],
            batch_size: int = 1,
            target_shape: tuple = (500, 500),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            split: float = None,
            nb_channel: int = 3,
            glob_pattern: str = './videos/{classname}/*.mp4',
            _validation_data: list = None):

        # should be only RGB or Grayscale
        assert nb_channel in (1,3)

        # shape size should be 2
        assert len(target_shape) == 2
        
        # only two phase types
        assert phase in ['training','predict']

        # split factor should be a proper value
        if split is not None:
            assert 0.0 < split < 1.0

        # be sure that classes are well ordered
        classes.sort()

        self.phase = phase
        self.face_detection = face_detection
        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nb_frames = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self._random_trans = []
        self.files = []
        self.validation = []


        if _validation_data is not None:
            
            # we only need to set files here
            self.files = _validation_data
        
        else:
            
            # Identify video files for training and validation
            if phase == 'training':
            
                if split is not None and split > 0.0:
                    
                    for i in classes:
                        
                        files = glob.glob(glob_pattern.format(classname=i))
                        n_files = len(files)
                        nbval = int(split * n_files)
                        nbtrain = n_files-nbval
                        print("class %s, train count: %d" % (i, nbtrain))
                        print("class %s, test count: %d" % (i, nbval))
    
                        # generate test indexes
                        indexes = np.arange(n_files)
                        if shuffle: np.random.shuffle(indexes)
                        
                        # get some sample
                        val = np.random.permutation(indexes)[:nbval]
                        
                        # remove test from train
                        indexes = np.array([i for i in indexes if i not in val])
    
                        # make the file lists
                        self.files += [files[i] for i in indexes]
                        self.validation += [files[i] for i in val]
    
                else:
                    
                    for i in classes:
                        
                        self.files += glob.glob(glob_pattern.format(classname=i))
            
            # Identify video files for prediction
            else:
                
                files = glob.glob(glob_pattern)
                n_files = len(files)
                print("new video count: %d" % n_files)
                indexes = np.arange(n_files)
                self.files += files

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        # to initialize transformations and shuffle indices
        self.on_epoch_end()

    def get_validation_generator(self):
        
        """ Return the validation generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nb_frames,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            _validation_data=self.validation)

    def on_epoch_end(self):
        
        """ Called by Keras after each epoch """
        if self.transformation is not None:
            
            self._random_trans = []
            for i in range(self.files_count):
                
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle and self.phase == 'training':
            
            np.random.shuffle(self.indexes)

    def __len__(self):
        
        return int(np.floor(self.files_count / self.batch_size))


    def detect_faces(self, frame):
        
        # Crop image to max-detected face
        frame_orig = frame
        try:
            
            dets = detector.run(frame,0,-0)
            shape = self.target_shape
            
            if len(dets[0]) == 0:
                
                frame = cv2.resize(frame, shape)
            
            else:
                
                max_face = dets[0][dets[1].index(max(dets[1]))]
                x = max_face.left()
                y = max_face.top()
                w = max_face.right() - x
                h = max_face.bottom() - y
                x -= 50; y -= 50; w += 100; h += 100
                if x < 0: x = 0
                if y < 0: y = 0
                frame = cv2.resize(frame[y:y+h,x:x+w], shape)
            
            return(frame)
        
        except:
            
            return(frame_orig)
        

    def __getitem__(self, index):
        
        classes = self.classes
        shape = self.target_shape
        nb_frames = self.nb_frames
        batch_size = self.batch_size
        face_detection = self.face_detection
        indexes = self.indexes[index*batch_size:(index+1)*batch_size]
        labels = []
        images = []
        transformation = None

        # Iterate over videos and process the frames
        for i in indexes:
            
            # prepare a transformation if provided
            if self.transformation is not None:
                transformation = self._random_trans[i]

            # video = random.choice(files)
            video = self.files[i]
            
            # Generate the target array of labels
            if self.phase == 'training':
                
                classname = video.split(os.sep)[-2]
                if len(classes) == 2:
                    # Assign label to be 0 or 1 for binary classification
                    label = classes.index(classname)
                else:
                    # create a label array and set 1 to the right column
                    # for multi-class classification
                    label = np.zeros(len(classes))
                    col = classes.index(classname)
                    label[col] = 1.
                
                labels.append(label)

            cap = cv2.VideoCapture(video)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame_step = floor(total_frames/nb_frames/2)
            frames = []
            frame_i = 0
            while True:
                
                # Read the next frame from the video
                grabbed, frame = cap.read()
                
                # if the frame was not grabbed, the end of the stream is reached
                if not grabbed: 
                    break
                
                # Account for corrupted videos
                if frame_step == 0:
                    break
                
                frame_i += 1
                if frame_i % frame_step == 0:
                    
                    # Convert to grayscale
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    if face_detection:
                        
                        # Try to isolate the max probability face
                        try:
                            frame = self.detect_faces(frame)
                        except:
                            frame = cv2.resize(frame, shape)
                            print("Face Detection Failure: "+video)
                            
                    else:
                        
                        frame = cv2.resize(frame, shape)
                
                    # Add processed frame to the sequence
                    frame = img_to_array(frame) * self.rescale
                    frames.append(frame)
                    
                    # Break once the appropriate number of frames is collected
                    if len(frames) == nb_frames:
                        break

            # End the video capture
            cap.release()

            # apply transformation
            if transformation is not None:
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # reshape the arrays into sequences for LSTM (timesteps, features)
            # (500*500*3 = 750000) (HEIGHT*WIDTH*CHANNELS)
            #frames = np.array(frames)
            #frames = frames.reshape(self.nb_frames,750000)
    
            # add the sequence in batch
            images.append(frames)                

        # Return the final image tensors and label arrays        
        return np.array(images), np.array(labels)
