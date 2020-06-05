"""
Standard image classification in TensorFlow
https://www.tensorflow.org/tutorials/images/classification

Build data input pipelines
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.handler_data_path import get_data_path


print("TensorFlow Version: {}".format(tf.__version__))

AUTOTUNE = tf.data.experimental.AUTOTUNE


data_dir = os.path.join(get_data_path(), "leaf-classification")
train_dir = os.path.join(data_dir, 'train_images')
validation_dir = os.path.join(data_dir, 'validation_images')

CLASS_NAMES = [x for x in sorted(os.listdir(str(train_dir))) if x[0] != '.']
CLASS_NAMES = np.array(CLASS_NAMES)
print("Number of classes: {}".format(len(CLASS_NAMES)))

image_count = 0
for class_ in [x for x in os.listdir(train_dir) if x[0] != '.']:
    for image_ in os.listdir(os.path.join(train_dir, class_)):
        image_count += 1
print("Image count:", image_count)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
