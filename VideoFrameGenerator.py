#!/usr/bin/env python

"""
VideoFrameGenerator - Simple Generator
--------------------------------------
A simple frame generator that takes distributed frames from
videos. It is useful for videos that are scaled from frame 0 to end
and that have no noise frames.

https://github.com/metal3d/keras-video-generators/blob/master/src/keras_video/generator.py
https://github.com/keras-team/keras/issues/12586
"""

import os
import glob
import numpy as np
import cv2
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

class VideoFrameGenerator(Sequence):
    
    """
    Create a generator that return batches of frames from video
    - rescale: float fraction to rescale pixel data (commonly 1/255.)
    - nb_frames: int, number of frames to return for each sequence
    - classes: list of str, classes to infer
    - batch_size: int, batch size for each loop
    - use_frame_cache: bool, use frame cache (may take a lot of memory for \
        large dataset)
    - shape: tuple, target size of the frames
    - shuffle: bool, randomize files
    - transformation: ImageDataGenerator with transformations
    - split: float, factor to split files and test
    - nb_channel: int, 1 or 3, to get grayscaled or RGB images
    - glob_pattern: string, directory path with '{classname}' inside that \
        will be replaced by one of the class list
    - _test_data: already filled list of data, **do not touch !**
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
            rescale=1/255.,
            nb_frames: int = 10,
            classes: list = [],
            batch_size: int = 1,
            use_frame_cache: bool = False,
            target_shape: tuple = (224, 224),
            shuffle: bool = True,
            transformation: ImageDataGenerator = None,
            split: float = None,
            nb_channel: int = 3,
            glob_pattern: str = './videos/{classname}/*.mp4',
            _test_data: list = None):

        # should be only RGB or Grayscale
        assert nb_channel in (1,3)

        # we should have classes
        assert len(classes) != 0

        # shape size should be 2
        assert len(target_shape) == 2

        # split factor should be a propoer value
        if split is not None:
            assert 0.0 < split < 1.0

        # be sure that classes are well ordered
        classes.sort()

        self.rescale = rescale
        self.classes = classes
        self.batch_size = batch_size
        self.nb_frames = nb_frames
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.nb_channel = nb_channel
        self.transformation = transformation
        self.use_frame_cache = use_frame_cache
        self._random_trans = []
        self.__frame_cache = {}
        self.files = []
        self.test = []

        if _test_data is not None:
            
            # we only need to set files here
            self.files = _test_data
        
        else:
            
            if split is not None and split > 0.0:
                
                for i in classes:
                    
                    files = glob.glob(glob_pattern.format(classname=i))
                    n_files = len(files)
                    nbval = int(split * n_files)
                    print("class %s, test count: %d" % (i, nbval))

                    # generate test indexes
                    indexes = np.arange(n_files)
                    if shuffle: np.random.shuffle(indexes)
                    
                    # get some sample
                    val = np.random.permutation(indexes)[:nbval]
                    
                    # remove test from train
                    indexes = np.array([i for i in indexes if i not in val])

                    # and now, make the file list
                    self.files += [files[i] for i in indexes]
                    self.test += [files[i] for i in val]

            else:
                
                for i in classes:
                    
                    self.files += glob.glob(glob_pattern.format(classname=i))

        # build indexes
        self.files_count = len(self.files)
        self.indexes = np.arange(self.files_count)
        self.classes_count = len(classes)

        # to initialize transformations and shuffle indices
        self.on_epoch_end()

        print("get %d classes for %d files for %s" % (
            self.classes_count,
            self.files_count,
            'train' if _test_data is None else 'test'))

    def get_test_generator(self):
        
        """ Return the test generator if you've provided split factor """
        return self.__class__(
            nb_frames=self.nb_frames,
            nb_channel=self.nb_channel,
            target_shape=self.target_shape,
            classes=self.classes,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            rescale=self.rescale,
            _test_data=self.test)

    def on_epoch_end(self):
        
        """ Called by Keras after each epoch """
        if self.transformation is not None:
            
            self._random_trans = []
            for i in range(self.files_count):
                
                self._random_trans.append(
                    self.transformation.get_random_transform(self.target_shape)
                )

        if self.shuffle:
            
            np.random.shuffle(self.indexes)

    def __len__(self):
        
        return int(np.floor(self.files_count / self.batch_size))

    def __getitem__(self, index):
        
        classes = self.classes
        shape = self.target_shape
        nb_frames = self.nb_frames
        labels = []
        images = []
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        transformation = None

        for i in indexes:
            
            # prepare a transformation if provided
            if self.transformation is not None:
                
                transformation = self._random_trans[i]

            # video = random.choice(files)
            video = self.files[i]
            classname = video.split(os.sep)[-2]

            # create a label array and set 1 to the right column
            label = np.zeros(len(classes))
            col = classes.index(classname)
            label[col] = 1.

            if video not in self.__frame_cache:
                
                cap = cv2.VideoCapture(video)
                frames = []
                while True:
                    
                    grabbed, frame = cap.read()
                    if not grabbed: 
                        break
                    
                    # resize to (500,500,3)
                    frame = cv2.resize(frame, shape)

                    # use RGB or Grayscale
                    if self.nb_channel == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    # to np
                    frame = img_to_array(frame) * self.rescale
                    frames.append(frame)

                # Add 2 frames to drop first and last frame
                jump = len(frames)//(nb_frames+2)

                # get only some images
                try:
                    frames = frames[jump::jump][:nb_frames]
                except Exception as exception:
                    print(video)
                    raise exception

                # add to frame cache to not read from disk later
                if self.use_frame_cache:
                    self.__frame_cache[video] = frames
            
            else:
                
                frames = self.__frame_cache[video]

            # apply transformation
            if transformation is not None:
                
                frames = [self.transformation.apply_transform(
                    frame, transformation) for frame in frames]

            # reshape the frames (500*500*3 = 750000)
            # (timesteps, features)
            frames = np.array(frames)
            #frames = frames.reshape(10,750000)
    
            # add the sequence in batch
            images.append(frames)
            labels.append(label)

        # Return the final image tensors and label arrays        
        return np.array(images), np.array(labels)
