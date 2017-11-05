# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 19:27:43 2017

@author: sraghunath

Plot shows the data cannot be linearly separated 
"""


import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)




svm = SVC(C=10.0,kernel='rbf',gamma=0.01,random_state=1)

svm.fit(X_xor,y_xor)