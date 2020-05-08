from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import cv2

import pandas as pd
import scipy.io as sio
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

data = pd.read_csv("data/training/solution.csv", index_col=0)
lable = np.asarray(data)
lable = lable-1

data_train = []
for i in range(1, 5001):
    m = cv2.imread('data/training/training/' + str(i) + '.png', 1)
    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    ret, m = cv2.threshold(m, 0, 1, cv2.THRESH_BINARY)
    m = cv2.bitwise_not(m)
    m = cv2.medianBlur(m, 3)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.dilate(m, kernel, iterations=1)
    m = cv2.medianBlur(m, 3)
    height = m.shape[0]
    width = m.shape[1]
    if (height < width):
        diff = int((width - height) / 2)
        m1 = cv2.copyMakeBorder(m, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=254)
    elif (height > width):
        diff = int((height - width) / 2)
        m1 = cv2.copyMakeBorder(m, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=254)
    else:
        m1 = m
    m1 = m1[60:m1.shape[0] - 60, 60:m1.shape[1] - 60]
    kernel = np.ones((3, 3), np.uint8)
    m1 = cv2.dilate(m1, kernel, iterations=1)
    # m1 = Image.fromarray(m1)
    # m1 = m1.resize((150, 150), Image.ANTIALIAS)
    # m1 = np.asarray(m1)
    kernel = np.ones((2, 2), np.uint8)
    m1 = cv2.dilate(m1, kernel, iterations=1)
    #     m1 = cv2.bitwise_not(m1)
    m1 = cv2.resize(m1, (150, 150))

    m1 = m1.reshape((150, 150, 1))
    if i % 500 == 0:
        print("Check : ", i)

    data_train.append(m1)

data_t = np.asarray(data_train)


X_t = data_t.reshape(5000,150,150,1)[0:4500]/255
y_t = lable[0:4500]

X_ts = data_t.reshape(5000,150,150,1)[4500:5000] /255
y_ts = lable[4500:5000]


visible = Input(shape=(150,150,1))
conv1 = Conv2D(16, kernel_size=4, activation='relu')(visible)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
drop1 = Dropout(0.5)(pool1)

conv3 = Conv2D(32, kernel_size=4, activation='relu')(drop1)
conv4 = Conv2D(32, kernel_size=4, activation='relu')(conv3)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
drop2 = Dropout(0.5)(pool2)

conv5 = Conv2D(64, kernel_size=4, activation='relu')(drop2)
conv6 = Conv2D(64, kernel_size=4, activation='relu')(conv5)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
drop3 = Dropout(0.5)(pool3)


flat = Flatten()(drop2)
hidden1 = Dense(512, activation='relu')(flat)
drop4 = Dropout(0.5)(hidden1)
hidden2 = Dense(16, activation='relu')(drop4)
output = Dense(6, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_t, y_t, epochs=5, verbose =1, validation_data=(X_ts, y_ts),batch_size= 32,shuffle=True,steps_per_epoch=None)