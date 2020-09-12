"""
Standard image classification in TensorFlow:
https://www.tensorflow.org/tutorials/images/classification

We have a trained model in tensorflow, we store it and we would like to pursue training on some more training.
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

print(tf.version.VERSION)

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = os.path.join(get_data_path(), "leaf-classification")
train_dir = os.path.join(data_dir, 'train_images')
validation_dir = os.path.join(data_dir, 'validation_images')

CLASS_NAMES = [x for x in sorted(os.listdir(str(train_dir))) if x[0] != '.']
CLASS_NAMES = np.array(CLASS_NAMES)
print("Number of classes: {}".format(len(CLASS_NAMES)))

# Loading the model

checkpoint_dir = "/Users/thibaultdesfontaines/data/training_1/"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("{}".format(latest))

image_count = 0
for class_ in [x for x in os.listdir(train_dir) if x[0] != '.']:
    for image_ in os.listdir(os.path.join(train_dir, class_)):
        image_count += 1
print("Image count:", image_count)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_image_generator = ImageDataGenerator(
                     rescale=1./255,
                     horizontal_flip=True,
                     rotation_range=360,
                     )

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
                            batch_size=BATCH_SIZE,
                            directory=train_dir,
                            shuffle=True,
                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                            classes=list(CLASS_NAMES),
                            class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(
                            batch_size=BATCH_SIZE,
                            directory=validation_dir,
                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                            classes=list(CLASS_NAMES),
                            class_mode='categorical')

sample_training_images, _ = next(train_data_gen)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(99)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

model.load_weights(latest)

total_train = 0
for class_ in os.listdir(train_dir):
    if class_[0] == ".":
        continue
    total_train += len(os.listdir(os.path.join(train_dir, class_)))

total_val = 0
for class_ in os.listdir(validation_dir):
    if class_[0] == ".":
        continue
    total_val += len(os.listdir(os.path.join(validation_dir, class_)))

epochs = 10

checkpoint_path = "/Users/thibaultdesfontaines/data/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

cp_earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // BATCH_SIZE,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // BATCH_SIZE,
    callbacks=[cp_callback, cp_earlystopping]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
