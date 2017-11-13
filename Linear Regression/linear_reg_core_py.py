# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:36:36 2017

@author: sraghunath

Linear Regression using Gradient Descent 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class LinearRegressionGD(object) : 
    
    def __init__(self,no_iters=20,learning_rate=0.001) : 
        
        self.learning_rate = learning_rate
        self.no_iters = no_iters
        
    def fit(self,X,y) :
        
        self.w_ = np.zeros(X.shape[1] + 1)
        
        self.cost_ = [] 
        
        for i in range(self.no_iters) : 
            
            errors = (y - self.get_output(X))
            
            self.w_[0] = self.learning_rate * errors.sum()
            
            self.w_[1:] = self.learning_rate * np.dot(X.T,errors)
            
            self.cost_.append(0.5 * (errors**2).sum())
            
        return self
    
    def get_output(self,X) : 
        
        return self.w_[0] + np.dot(self.w_[1:],X.T)
    
    def predict(self,X_test) : 
        
        return self.get_output(X_test)
    
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-2nd-edition'
                 '/master/code/ch10/housing.data.txt',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['RM']].values

y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_std = sc.fit_transform(X)

y_std = sc.fit_transform(y[:,np.newaxis]).flatten()

lmodel = LinearRegressionGD()

lmodel.fit(X_std,y_std)

plt.plot(range(1,lmodel.no_iters+1),lmodel.cost_)

plt.xlabel("Number of epochs")
plt.ylabel("SSE Cost Function")
plt.show()

def linear_plot(X,y,model) :
    
    plt.scatter(X,y,c="steelblue")
    plt.plot(X,model.predict(X),color="black")


    


    
    

        
    
    