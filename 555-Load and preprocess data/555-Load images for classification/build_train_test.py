"""
Build the distinctive folders for training and testing of the model
"""

import os
import pandas as pd
import pdb
import shutil

from utils.handler_data_path import get_data_path

data_dir = os.path.join(get_data_path(), "leaf-classification")
image_dir = os.path.join(data_dir, "images")

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

test_dir = os.path.join(data_dir, "test_images")
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

test_index = []
for row_index in range(len(test_df)):
    row = test_df.iloc[row_index]
    picture_id = int(row['id'])
    test_index.append(picture_id)

for index_ in test_index:
    source_img_path = os.path.join(image_dir, "{}.jpg".format(index_))
    target_img_path = os.path.join(test_dir, "{}.jpg".format(index_))
    shutil.copy(source_img_path, target_img_path)



