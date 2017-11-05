# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:22:13 2017

@author: sraghunath
"""

# demonstrates how to calculate the cross-entropy error function
# in numpy.

import numpy as np

N = 100
D = 2


X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)
