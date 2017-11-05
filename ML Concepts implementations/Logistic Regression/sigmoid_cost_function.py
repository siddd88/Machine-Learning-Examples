# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:12:15 2017

@author: sraghunath


Learning the weights of the logistic cost function


"""



import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z) : 
    return 1.0/(1.0 + np.exp(-z))

def cost_1(z) :     #cost if y = 1 
    return -np.log(sigmoid(z))

def cost_0(z) : # cost if y = 0
    return - np.log(1 - sigmoid(z))


z = np.arange(-7,7,0.1)

phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
c0 = [cost_0(x) for x in z]

plt.plot(phi_z,c1,label="Cost (J(w)) if y =1 ",linestyle="--")
plt.plot(phi_z,c0,label="Cost (J(w)) if y =0 ")

plt.xlabel("Sigmoid(z)")
plt.ylabel("Cost J(w)")

plt.legend(loc="best")

plt.show()

