# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:01:34 2017

@author: sraghunath
"""

from sklearn import datasets
import numpy as np
from logistic_reg_core_py import LogisticRegressionGD
import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv("ecommerce_data.csv")

df = df.query("user_action==1 or user_action==0")

df.head()

#Normalilizing columns  n_products_viewed and visit_duration


df['n_products_viewed'] =(df['n_products_viewed'] - df['n_products_viewed'].mean()) / df['n_products_viewed'].std()

df['visit_duration'] =(df['visit_duration'] - df['visit_duration'].mean()) / df['visit_duration'].std()

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values

lmodel = LogisticRegressionGD()

lmodel.fit(X,y)


