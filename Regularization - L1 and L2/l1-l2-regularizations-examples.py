# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:48:01 2017

@author: sraghunath
"""
import pandas as pd 
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)



df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))

df_wine.head()

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =\
    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)
    
    
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)
print('Training accuracy (L1):', lr.score(X_train_std, y_train))
print('Test accuracy (L1):', lr.score(X_test_std, y_test))



LogisticRegression(penalty='l2')

from sklearn.linear_model import LogisticRegression

lr2 = LogisticRegression(penalty='l2', C=1.0)
lr2.fit(X_train_std, y_train)
print('Training accuracy (L2):', lr2.score(X_train_std, y_train))
print('Test accuracy (L2):', lr2.score(X_test_std, y_test))
