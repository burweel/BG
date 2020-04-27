# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 06:51:27 2020

@author: burwe
"""

#insurance dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
first_df = pd.read_csv('insurance2.csv')
second_df = pd.read_csv('insurance3r2.csv')

# в датасете second нет признака steps, избавляемся от него, чтобы была возможность сравнивать

second_df = second_df.drop(columns='steps')

# проектирование признаков
sex_dumm = pd.get_dummies(first_df['sex']).rename({0: "female", 1: "male"}, axis=1)
region_dumm = pd.get_dummies(first_df['region'], prefix='region')
first_df = first_df.drop(columns=['sex', 'region'])
first_df['ideal_bmi'] = first_df['bmi'].apply(lambda x: 1 if 18.5 < x < 25 else 0)
first_df = pd.concat([first_df, sex_dumm, region_dumm], axis=1)

y = first_df['insuranceclaim']
X = first_df.drop(columns='insuranceclaim')

#first_df.isnull().sum() == 0, NaN в датасете нет

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42,
                                                        test_size=0.3)

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

linear = SGDClassifier(tol=1e-6)
linear.fit(X_train, y_train)
acc_linear = linear.score(X_test, y_test) # 0.823

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
acc_forest = forest.score(X_test, y_test) # 0.9054

params = {'alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
          'penalty': ["l1", "l2"]}

grid = GridSearchCV(SGDClassifier(), params, cv=5)
grid.fit(X_train, y_train)
acc_grid = grid.score(X_test, y_test)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2, interaction_only=True)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, 
                                                                        test_size=0.3,
                                                                        random_state=42)

lin_poly = SGDClassifier()
lin_poly.fit(X_train_poly, y_train_poly)
acc_poly = lin_poly.score(X_test_poly, y_test_poly)

forest_poly = RandomForestClassifier().fit(X_train_poly, y_train_poly)
acc_forest_poly = forest_poly.score(X_test_poly, y_test_poly)

# Нелинейные признаки ИНОГДА дают лучшие значения
# нужно добавить pipeline и проверить модель на second_df
