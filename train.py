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

print("Compiling model..")
opt = Adam(lr=BASE_LEARNING_RATE, decay=BASE_LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# fine-tune model
print("Fine-tuning model")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, trainX),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS)

# Eval model
print("Evaluating model..")
pred = model.predict(testX, batch_size=BATCH_SIZE)

# find index of label with largest predicted probability
pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred,
    target_names=lb.classes_))

# confusion matrix
cm = confusion_matrix(testY.argmax(axis=1), pred)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("accuracy: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# Plot the loss and accuracy
eps = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, eps), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, eps), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, eps), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, eps), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
print("Saving COVID-19 model..")
model.save(args["model"], save_format="h5")
print("Finished!")