"""
Loading CSV data into TensorFlow:
https://www.tensorflow.org/tutorials/load_data/numpy
"""

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from utils.handler_data_path import get_data_path
import pdb

print("TensorFlow Version: {}".format(tf.__version__))

data_dir = os.path.join(get_data_path(), "santander_customer_transaction_prediction")
print('[+] data_dir={}'.format(data_dir))

train_file_path = os.path.join(data_dir, "train.csv")

df = pd.read_csv(train_file_path)

split_index = int(len(df)*0.75)

train_df = df[:split_index]
validation_df = df[split_index:]

train_examples = train_df[['var_{}'.format(i) for i in range(200)]].to_numpy()
validation_examples = validation_df[['var_{}'.format(i) for i in range(200)]].to_numpy()
train_labels = train_df['target']
validation_labels = validation_df['target']

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_examples, validation_labels))

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, epochs=10, validation_data=validation_dataset)


model.evaluate(validation_dataset)
