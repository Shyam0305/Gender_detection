import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Activation,Flatten,Dropout,Dense,BatchNormalization
from tqdm.notebook import tqdm

BASE_DIR = "UTKFace"

image_paths = []
image_data = []
gender_labels = []

image_files = os.listdir(BASE_DIR)
random.shuffle(image_files)

for image in image_files:
    image_path = os.path.join(BASE_DIR,image)
    img = image.split("_")

    img1 = cv2.imread(image_path)

    if img1 is not None:
        img1 = cv2.resize(img1,(96,96))
        img1 = img_to_array(img1)
        image_data.append(img1)
        
        gender_labels.append(int(img[1]))
        image_paths.append(image_path)


  
gender_mapping = {
    0 : 'Male',
    1 : 'Female'
}

image_data = np.array(image_data)
gender_labels = np.array(gender_labels)

image_data = image_data/255.0

(trainX,testX,trainY,testY) = train_test_split(image_data,gender_labels,test_size=0.2,random_state=42)

trainY_one = to_categorical(trainY,num_classes=2)
testY_one = to_categorical(testY,num_classes=2)

aug = ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

# trainY.shape
# trainY_one.shape


def build_model(width,height,depth,classes):
  model = Sequential()

  input_shape = (height,width,depth)
  chan_dim = -1

  if K.image_data_format() == "channels_first":
    input_shape = (depth,height,width)
    chan_dim = 1

  
  model.add(Conv2D(32,(3,3),padding="same",input_shape=input_shape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chan_dim))
  model.add(MaxPooling2D(pool_size=(3,3)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chan_dim))

  model.add(Conv2D(64,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chan_dim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(3,3),padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chan_dim))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(classes))
  model.add(Activation("sigmoid"))

  return model

model = build_model(96,96,3,2)

# opt = Adam(1e-3,1e-3/50)
model.compile(loss="binary_crossentropy",optimizer='Adam',metrics=["accuracy"])

# trainX.shape
# trainY_one.shape

res = model.fit(aug.flow(trainX,trainY_one,batch_size=64),validation_data=(testX,testY_one),steps_per_epoch=len(trainX)//64,epochs=50,verbose=1)
model.save('gender_detection.keras')