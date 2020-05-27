from utils.handler_data_path import get_data_path

from _collections import defaultdict
import IPython.display as display
from PIL import Image
import numpy as np
import os
import pathlib
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import shutil
import tqdm
import tensorflow as tf

print("TensorFlow Version: {}".format(tf.__version__))

data_dir = get_data_path() + "leaf-classification/"
image_dir = data_dir + "images/"
print("Images dir: {}".format(image_dir))

image_dir = pathlib.Path(image_dir)
image_count = len(list(image_dir.glob('*.jpg')))
print("Detected {} images".format(image_count))

train_df = pd.read_csv(data_dir+"train.csv")
test_df = pd.read_csv(data_dir+"test.csv")
sample_submission_df = pd.read_csv(data_dir+"sample_submission.csv")

species = np.unique(train_df['species'])
assert len(species) == 99
CLASS_NAMES = species

# Group id by species
groupby_species = defaultdict(list)
for i in range(len(train_df)):
    row = train_df.iloc[i]
    specie_, id_ = row['species'], row['id']
    groupby_species[specie_].append(id_)

new_folder = os.path.join(data_dir, "new_images")
if not os.path.exists(new_folder):

    os.mkdir(new_folder)

    # Redispatch images in folder
    for class_ in tqdm.tqdm(CLASS_NAMES):
        subfolder = os.path.join(data_dir, "new_images", class_)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        for id_ in groupby_species[class_]:
            file_name_ = "{}.jpg".format(id_)
            src = os.path.join(str(image_dir), file_name_)
            target = os.path.join(subfolder, file_name_)
            shutil.copy(src, target)

