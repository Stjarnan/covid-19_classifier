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

# Init base hyperparameters
BASE_LEARNING_RATE = 1e-3
EPOCHS = 25
BATCH_SIZE = 8

# Load dataset and create label and data lists
print("Loading dataset..")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over paths,
#   extract label and data
for imagePath in imagePaths:
    
    label = imagePath.split(os.path.sep)[-2]

    # swap channels & resize to a fixed size
    # (our CNN wants 244x244 inputs)
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.color_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # update data and label list
    data.append(image)
    labels.append(label)

# scale pixel intensities of the images
# convert both lists to Numpy arrays
data = np.array(data) / 255
labels = np.array(labels)

# label OH-encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.2, stratify=labels, random_state=1)

# augment data
aug = ImageDataGenerator(rotation_range=10, fill_mode="nearest")

# load VGG16 model (for fine-tuning)
vgg16 = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# new top for the model to fine-tune on
model_top = vgg16.output
model_top = AveragePooling2D(pool_size=(4,4))(model_top)
model_top = Flatten(name="flatten")(model_top)
model_top = Dense(64, activation="relu")(model_top)
model_top = Dropout(0.5)(model_top)
model_top = Dense(2, activation="softmax")(model_top)

# Connect vgg16 with new top
model = Model(inputs=vgg16.input, outputs=model_top)

# Freeze layers from vgg16
for layer in vgg16.layers:
    layer.trainable = False