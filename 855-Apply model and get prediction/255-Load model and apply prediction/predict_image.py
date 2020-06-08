"""

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import pdb

from utils.handler_data_path import get_data_path

from PIL import Image


print(tf.version.VERSION)

data_dir = os.path.join(get_data_path(), "leaf-classification")
train_dir = os.path.join(data_dir, 'train_images')
validation_dir = os.path.join(data_dir, 'validation_images')
test_dir = os.path.join(data_dir, 'test_images')

CLASS_NAMES = [x for x in sorted(os.listdir(str(train_dir))) if x[0] != '.']
CLASS_NAMES = np.array(CLASS_NAMES)
print("Number of classes: {}".format(len(CLASS_NAMES)))


checkpoint_dir = "/Users/thibaultdesfontaines/data/training_1/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("{}".format(latest))

IMG_HEIGHT = 224
IMG_WIDTH = 224


def create_model():

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(99)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = create_model()

model.summary()

model.load_weights(latest)

for image_path in os.listdir(test_dir):
    image_abs_path = os.path.join(test_dir, image_path)
    img = Image.open(image_abs_path)
    img_array = np.array(img.resize((IMG_WIDTH, IMG_HEIGHT)))
    break

