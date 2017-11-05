# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:51:22 2017

@author: sraghunath
"""

## Using the iris Dataset which is available in Sklearn API 

from sklearn import datasets
import numpy as np
from logistic_reg_core_py import LogisticRegressionGD
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, [2, 3]] #Petal length and the petal width
y = iris.target


from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=1,stratify=y)

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from plot_decision_regions import plot_decision_regions

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_05.png', dpi=300)
plt.show()