# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:16:08 2020

@author: burwe
"""

# задача спрогнозировать последовательность с keras
# подключаем библиотеки
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import keras

import matplotlib.pyplot as plt

path = 'GOOGL_2006-01-01_to_2018-01-01.csv'
df = pd.read_csv(path)

#переводим в формат времени
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
#переводим время в индексы
df.set_index('Date', inplace=True)
df.drop("Name", axis=1, inplace=True)
scaler = MinMaxScaler()
df_1 = scaler.fit_transform(df)
df_1 = pd.DataFrame(df_1, columns=df.columns)

# приступаем к построению модели
input_len = 30
test = df_1[df.index >= "2017-01-01"]
train = df_1[df.index < "2017-01-01"]

# Составим обучающую и тестовую выборку, цель - предсказать переменную High
# на участке 2768-3018. Обучение проводится на участке 0-2767. Обучающий массив
# - переменная High с шириной окна 30.

y_train = train["High"] 
y_test = test["High"]
X_train = train.drop(columns="High")
X_test = test.drop(columns="High")

train_x = []
for i in range(input_len, len(train)):
    train_x.append(X_train.iloc[i-30:i].values)

train_y = y_train.iloc[input_len:].values

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
for i in range(input_len, len(test)):
    test_x.append(X_test.iloc[i-30:i].values)

test_y = y_test.iloc[input_len:].values

test_x = np.array(test_x)
test_y = np.array(test_y)

# начинаем собирать модель, включая рекуррентные слои
model = keras.Sequential()  
model.add(keras.layers.LSTM(10, input_shape=(30, 4)))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Dense(1))

model.compile(optimizer="RMSprop", loss="MSE")

model.fit(train_x, train_y, epochs=5)
results = model.predict(test_x).reshape(test_y.shape[0])
mse = mean_squared_error(test_y, results)

# модель обучается ~20 sec
# имеется возможность для дальнейшего совершенствования модели,
# подбора других слоев и функции потерь
# mse = 0.000909662