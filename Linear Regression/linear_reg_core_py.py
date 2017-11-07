# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:36:36 2017

@author: sraghunath

Linear Regression using Gradient Descent 

"""
import numpy as np

class LinearRegressionGD(object) : 
    
    def __init__(self,learning_rate=0.001,epochs) : 
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self,X,y) : 
        
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []
        
        for i in range(self.epochs) :
            
            output = self.net_input(X)
            errors = (y-output)
            cost =  (errors**2).sum()*0.5
            self.w_[1:] = self.learning_rate * X.T.dot(errors)
            self.w_[0] = self.learning_rate * errors.sum()
            
        return self
    
    def net_input(self,X) : 
        
        return np.dot(self.w_[1:],X) + self.w_[0]
    
    def predict(self,X_test): 
        
        return self.net_input(X_test)
    
    