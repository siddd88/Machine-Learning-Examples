# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:52:45 2017

@author: sraghunath
"""
import numpy as np

class LogisticRegressionGD(object) : 
    
    def __init__(self,learning_rate=0.01,epochs=15,random_state=1) : 
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state= random_state
        
    def fit(self,X_train,y_train) : 
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(size = 1 + X_train.shape[1])
        self.cost_ = []
        
        for i in range(self.epochs) : 
            
            phi_z = self.sigmoid(self.net_input(X_train))
            
            errors = phi_z - y_train
            
            self.w_[1:] += self.eta * X_train.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            cost_ = - np.dot(y_train,np.log(phi_z)) - (np.dot((1-y_train),np.log(1 - phi_z)))
            self.cost_.append(cost_)
            
        
        return self 
    
    def sigmoid(self,z) : 
        return 1.0 / (1 + np.exp(-z))
    
    def net_input(self,X) : 
        return np.dot(self.w_[1:],X) + self.w_[0]
    
    def predict(self,X_test) : 
        return np.where(self.sigmoid(self.net_input(X_test))>0.5,1,0)
    
    
        