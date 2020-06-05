import IPython.display as display
from PIL import Image
import os
import pathlib
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pdb

print("TensorFlow Version: {}".format(tf.__version__))

AUTOTUNE = tf.data.experimental.AUTOTUNE

image_dir = "/Users/thibaultdesfontaines/data/leaf-classification/new_images"
image_dir = pathlib.Path(image_dir)

CLASS_NAMES = [x for x in sorted(os.listdir(str(image_dir))) if x[0] != '.']
CLASS_NAMES = np.array(CLASS_NAMES)
print("Number of classes: {}".format(len(CLASS_NAMES)))

image_count = 0
for class_ in [x for x in os.listdir(image_dir) if x[0]!='.']:
    for image_ in os.listdir(os.path.join(image_dir, class_)):
        image_count += 1
print("Image count:", image_count)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(
        directory=str(image_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes=list(CLASS_NAMES))


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def process_test_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


list_ds = tf.data.Dataset.list_files(str(image_dir/'*/*'))
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))
train_images = tf.squeeze(image_batch)
# images_batch is TensorShape([32, 224, 224, 1])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(224, 224)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(99)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, tf.where(label_batch)[:, 1:], epochs=20)

# Test on new images

test_dir = pathlib.Path("/Users/thibaultdesfontaines/data/leaf-classification/test_images")

test_data_gen = image_generator.flow_from_directory(
        directory=str(test_dir),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes=list(CLASS_NAMES))

test_ds = tf.data.Dataset.list_files(str(test_dir/'*'))
test_ds = test_ds.map(process_test_path, num_parallel_calls=AUTOTUNE)
test_ds = prepare_for_training(test_ds)

test_batch = next(iter(test_ds))
test_batch = tf.squeeze(test_batch)

predictions = np.array(model.apply(test_batch))
