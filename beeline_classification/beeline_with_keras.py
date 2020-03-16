# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:07:22 2020

@author: burwe
"""

#beeline with keras

import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('train.csv')

y = train_df['y']
y = keras.utils.to_categorical(y)
drop_one = ['y']

for line in train_df:
    if train_df[line].dtype == 'object':
        drop_one.append(line)

train_df = pd.get_dummies(train_df[drop_one])

X_check = train_df.drop(columns='y')
y_check = train_df['y']
X_check = X_check.apply(lambda x: x.fillna(x.mean()),axis=0)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier().fit(X_check, y_check)
d_imp = pd.DataFrame([forest.feature_importances_]).T
d_imp.index = X_check.columns
d_imp = d_imp.sort_values(0)
d_imp = d_imp[::-1]
d_imp[0] = np.cumsum(d_imp[0])

drop_two=[]

drop_features = d_imp[d_imp[0] >= 0.9].index
for feature in drop_features:
    drop_two.append(feature)

X = train_df.drop(columns=drop_two)

X = X.apply(lambda x: x.fillna(x.mean()),axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=17)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = keras.models.Sequential()
model.add(keras.layers.Dense(256, input_shape=(X_train_scaled.shape[1],)))
model.add(keras.layers.Dropout(0.02))

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.01))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16))

model.add(keras.layers.Dense(7))
model.compile(optimizer="RMSprop", loss="MSE", metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=5)
results = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, results)
score, acc = model.evaluate(X_test_scaled, y_test)

#acc = 0.993466, t= ~40 sec
# удачно выбрана стратегия one_hot_encoding, посредством RandomForest
# отброшены признаки, которые влияют менее чем на 0.9 суммарно на результат.
# есть возможность дальнейшей проработки модели и приведения кода к красивому виду
