# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 19:27:43 2017

@author: sraghunath

Transforms the data onto a higher dimension and projects it back using the linear trick via RBF Kernel
"""


import matplotlib.pyplot as plt
import numpy as np
from plot_data import plot_decision_regions

from sklearn.svm import SVC

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)




svm = SVC(C=10.0,kernel='rbf',gamma=0.01,random_state=1)

svm.fit(X_xor,y_xor)

plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_14.png', dpi=300)
plt.show()