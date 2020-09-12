"""
Loading CSV data into TensorFlow:
https://www.tensorflow.org/tutorials/load_data/csv
"""

import functools
import numpy as np
import os
import tensorflow as tf
from utils.handler_data_path import get_data_path
import pdb

import pandas as pd

np.set_printoptions(precision=3, suppress=True)

print("TensorFlow Version: {}".format(tf.__version__))

data_dir = os.path.join(get_data_path(), "santander_customer_transaction_prediction")
print('[+] data_dir={}'.format(data_dir))

LABEL_COLUMN = 'target'
LABELS = [0, 1]

train_file_path = os.path.join(data_dir, "train.csv")

df = pd.read_csv(train_file_path)


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
    return dataset


raw_train_data = get_dataset(train_file_path)


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))


show_batch(raw_train_data)

print("\n\n ------------")


SELECT_COLUMNS = ['target', 'var_0', 'var_1']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)

show_batch(temp_dataset)

print("\n\n ------------")

# Remove the User Id column in the dataset
SELECT_COLUMNS = ['target']+['var_{}'.format(i) for i in range(200)]
DEFAULTS = [0] + [0.0 for i in range(200)]
temp_dataset = get_dataset(train_file_path,
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)

print("\n\n ------------")

example_batch, labels_batch = next(iter(temp_dataset))


def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


packed_dataset = temp_dataset.map(pack)


for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())

show_batch(raw_train_data)

print("\n\n ------------")

example_batch, labels_batch = next(iter(temp_dataset))


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        return features, labels


NUMERIC_FEATURES = ['var_{}'.format(i) for i in range(200)]

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)

example_batch, labels_batch = next(iter(packed_train_data))

import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std


normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]


pdb.set_trace()

