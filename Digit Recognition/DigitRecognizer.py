# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:18:30 2019

@author: Navnit Singh
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Convolution2D,MaxPooling2D
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test)=mnist.load_data()

x_train.shape
y_train.shape

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

x_train=x_train/255
x_test=x_test/255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

model=Sequential()
model.add(Convolution2D(32,(5,5),input_shape=(1,28,28), activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=200, epochs=10, validation_data=(x_test,y_test))
scores=model.evaluate(x_test, y_test, verbose=0)
print(100-scores[1]*100)


