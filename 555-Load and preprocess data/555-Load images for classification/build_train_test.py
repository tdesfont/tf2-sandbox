"""
Build the distinctive folders for training and testing of the model
"""

import os
import pandas as pd
import pdb
import shutil
from collections import defaultdict
import tqdm
from utils.handler_data_path import get_data_path
import numpy as np

data_dir = os.path.join(get_data_path(), "leaf-classification")
image_dir = os.path.join(data_dir, "images")

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

train_dir = os.path.join(data_dir, "train_images")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

test_dir = os.path.join(data_dir, "test_images")
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

validation_dir = os.path.join(data_dir, "validation_images")
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)

groupby_species = defaultdict(list)
for i in range(len(train_df)):
    row = train_df.iloc[i]
    specie_, id_ = row['species'], row['id']
    groupby_species[specie_].append(id_)

CLASS_NAMES = np.unique(train_df['species'])

for class_ in tqdm.tqdm(CLASS_NAMES):
    train_class_folder = os.path.join(train_dir, class_)
    validation_class_folder = os.path.join(validation_dir, class_)
    if not os.path.exists(train_class_folder):
        os.mkdir(train_class_folder)
    if not os.path.exists(validation_class_folder):
        os.mkdir(validation_class_folder)
    j = 0
    for id_ in groupby_species[class_]:
        file_name = "{}.jpg".format(id_)
        src = os.path.join(str(image_dir), file_name)
        j += 1
        if j > 2:
            target = os.path.join(train_class_folder, file_name)
        else:
            target = os.path.join(validation_class_folder, file_name)
        shutil.copy(src, target)

for row_index in range(len(test_df)):
    row = test_df.iloc[row_index]
    picture_id = int(row['id'])
    source_img_path = os.path.join(image_dir, "{}.jpg".format(picture_id))
    target_img_path = os.path.join(test_dir, "{}.jpg".format(picture_id))
    shutil.copy(source_img_path, target_img_path)
