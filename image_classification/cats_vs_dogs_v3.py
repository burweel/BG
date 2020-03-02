# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:45:48 2020

@author: burwe
"""

#cats_vs_dogs
#import os, shutil

from keras import layers
from keras import models

#import os, shutil
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error

train_cats = "cats_vs_dogs\\train_dir\\train_cats_dir\\"
train_dogs = "cats_vs_dogs\\train_dir\\train_dogs_dir\\"
test_cats = "cats_vs_dogs\\test_dir\\test_cats_dir\\"
test_dogs = "cats_vs_dogs\\test_dir\\test_dogs_dir\\"

def get_arr(name):
    return Image.open(name).resize((150, 150))


train_cats_arr = np.array([np.array(get_arr(train_cats+"cat.{}.jpg".format(i))) 
                        for i in range(1000)])
train_dogs_arr = np.array([np.array(get_arr(train_dogs+"dog.{}.jpg".format(i))) 
                        for i in range(1000)])
test_cats_arr = np.array([np.array(get_arr(test_cats+"cat.{}.jpg".format(i))) 
                        for i in range(1000, 1500)])
test_dogs_arr = np.array([np.array(get_arr(test_dogs+"dog.{}.jpg".format(i))) 
                        for i in range(1000, 1500)])

train_cats_arr = np.true_divide(train_cats_arr, 255)
train_dogs_arr = np.true_divide(train_dogs_arr, 255)
test_cats_arr = np.true_divide(test_cats_arr, 255)
test_dogs_arr = np.true_divide(test_dogs_arr, 255)

X_train = np.vstack((train_cats_arr, train_dogs_arr))
y_train = np.hstack((np.ones(1000), np.zeros(1000)))
X_test = np.vstack((test_cats_arr, test_dogs_arr))
y_test = np.hstack((np.ones(500), np.zeros(500)))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(150, 150, 3)))
    
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(1e-4), 
              metrics=['acc'])


history = model.fit(X_train, y_train, epochs=20)
results = model.predict(X_test)
mse = mean_squared_error(y_test, results)

# mse = 0.18677107340613822, t= ~90 sec
# есть возможность дальнейшей проработки слоев модели, 
# возможность использования generator на обучении и проверке 