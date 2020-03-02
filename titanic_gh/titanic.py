# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:26:05 2020

@author: burwe
"""

#titanic
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
gs = pd.read_csv('gender_submission.csv')


# drop columns
list_to_drop = ['PassengerId', 'Ticket', 'Cabin']
df_train = df_train.drop(columns=list_to_drop)
df_test = df_test.drop(columns=list_to_drop)

# change name feature
df_train['len_of_name'] = [len(i.split()) for i in df_train['Name']]
df_test['len_of_name'] = [len(i.split()) for i in df_test['Name']]
df_train = df_train.drop(columns='Name')
df_test = df_test.drop(columns='Name')

#change sex feature
df_train = pd.concat([df_train, pd.get_dummies(df_train.Sex)], axis=1)
df_test = pd.concat([df_test, pd.get_dummies(df_test.Sex)], axis=1)
df_train = df_train.drop(columns='Sex')
df_test = df_test.drop(columns='Sex')

# change embarked feature 
df_train = pd.concat([df_train, pd.get_dummies(df_train.Embarked)], axis=1)
df_test = pd.concat([df_test, pd.get_dummies(df_test.Embarked)], axis=1)
df_train = df_train.drop(columns='Embarked')
df_test = df_test.drop(columns='Embarked')


# in catecategorical_features is nothing left
categorical_features = [i for i in df_train.columns if df_train[i].dtype == 'object']
number_features = [i for i in df_train.columns if df_train[i].dtype != 'object']

# fill NaN
df_train = df_train.apply(lambda x: x.fillna(x.mean()), axis=0)
df_test = df_test.apply(lambda x: x.fillna(x.mean()), axis=0)

y_train = df_train.Survived
X_train = df_train.drop(columns='Survived')
y_test = gs.Survived
X_test = df_test

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

forest = RandomForestClassifier()
lin_model = SGDClassifier()

forest.fit(X_train_scaled, y_train)
forest_acc = forest.score(X_test_scaled, y_test)

parameters = {'alpha': [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000]}

clf = GridSearchCV(lin_model, parameters, cv=5)
clf.fit(X_train_scaled, y_train)
acc = clf.score(X_test_scaled, y_test)
print(acc, clf.best_params_)

# очень хорошие результаты, accuracy = 0.985 для lin_model
# forest_acc хуже
