# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:01:06 2020

@author: burwe
"""

#dress recognition

from keras.datasets import fashion_mnist
from keras import models, layers
from keras.utils import to_categorical

import numpy as np
from sklearn.metrics import mean_squared_error

#dataset = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Conv2D(32, (2,2), input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(32, (2,2), input_shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(10))
model.compile(optimizer="RMSprop", metrics=['acc'], loss='mse')

model.fit(X_train, y_train, epochs=10)
results = model.predict(X_test)
mse = mean_squared_error(y_test, results)
acc = model.history.history['acc']
# accuracy = ~0.88, time = ~150 s
