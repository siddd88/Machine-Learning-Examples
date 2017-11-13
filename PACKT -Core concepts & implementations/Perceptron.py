# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:40:08 2017

@author: sraghunath
"""
import pandas as pd
import numpy as np 

class Perceptron(object) : 
    
    def __init__(self,learning_rate=0.01,n_iterations = 10,random_state=0) : 
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        
    def fit(self,X,y) : 
        
        random_obj = np.random.RandomState(self.random_state)
        
        self.w_ = random_obj.normal(size = X.shape[1] + 1)
        
        self.errors_ = []
        
        for  _ in range(self.n_iterations) : 
            
            misclassifications = 0
            
            for xi,yi in zip(X,y) : 
                
                self.w_[0] += self.learning_rate * (yi-self.predict(xi))
                
                self.w_[1:] += self.learning_rate * (yi-self.predict(xi)) * xi
                
                misclassifications += int(yi != self.predict(xi))    
                
            self.errors_.append(misclassifications)
            
            
        return self
    
    
    def net_input(self,X) : 
       
       return self.w_[0] + np.dot(self.w_[1:],X)
   
    def predict(self,X) : 
        
        return np.where(self.net_input(X) >= 0.0,1,-1)
    



    
