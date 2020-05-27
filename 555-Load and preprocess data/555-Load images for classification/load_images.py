from utils.handler_data_path import get_data_path

import IPython.display as display
from PIL import Image
import os
import pathlib
import tensorflow as tf

print("TensorFlow Version: {}".format(tf.__version__))

data_dir = get_data_path() + "leaf-classification/"
image_dir = os.path.join(data_dir, "new_images")
image_dir = pathlib.Path(image_dir)
print("Images dir: {}".format(image_dir))

CLASS_NAMES = [x for x in sorted(os.listdir(str(image_dir))) if x[0] != '.']
print("Number of classes: {}".format(len(CLASS_NAMES)))

sample_class = "Quercus_Trojana"
class_dir = os.path.join(image_dir, sample_class)
items = list(os.listdir(class_dir))

for image_path in items[:3]:
    display.display(Image.open(os.path.join(class_dir, image_path)))

