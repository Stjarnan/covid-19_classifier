import numpy as np
import argparse
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# collect arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True,
                    help="Path to dataset folder")
parser.add_argument("-p", "--plotpath", type=str, default="plot.png",
                    help="Path to output plot, include filename")
parser.add_argument("-m", "--model", default="covid19.model",
                    help="Path to output model, include filename")
args = vars(parser.parse_args())