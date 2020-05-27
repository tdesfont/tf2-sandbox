import os
import tensorflow as tf

DATA_PATH = "/home/tdesfont/Documents/data/leaf-classification/"

if __name__ == "__main__":
    print("TensorFlow Version: {}".format(tf.__version__))
    print(os.listdir(DATA_PATH))
    image_names = os.listdir(DATA_PATH+"images")
    print(image_names)