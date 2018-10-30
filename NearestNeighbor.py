# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:21:38 2018

@author: Dell
"""

import numpy as np
from collections import Counter

class NearestNeighbor():
    def __init__(self):
        print("")
    
    # Saves images values x and the labels y
    def train(self, x,y):
        self.Xtr = x            # images matrix
        self.Ytr = y            # images labels
    
    
    """ Calculates distance between the input images X and training data Xtr
    returns the majority vote of k neighbors """
    def predict(self, X , k=1, l='L1'):
        
        # number of test images
        num_test = X.shape[0]    
          
        # Initialize output matrix to zeros
        Ypred    = np.zeros(num_test, dtype = self.Ytr.dtype) 
                
        # Loop over input test images  matrix X
        for i in range (num_test):
           # if (l == 'L1'):    # Manhattan distance
               # distance between all the training data and test image X[i]
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis =1)
            l = self.Xtr - X[i,:]
           # else:              # Euclidean distance
            #   distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:])), axis = 1)
            
            # Indicies of the nearest k points 
            min_indicies = np.argsort(distances)[:k]
              # Votes of the k nearest neighbors
            votes = [self.Ytr[min_indicies[j]].tolist() for j in range(k)]

            
            """ res is list of tuples [('vote', n)]. vote is the maj. vote. n is the number of the maj. votes """
            # returns the most common vote in votes list
            res = Counter(votes).most_common(1)
            # Predicted y is the most commont vote
            Ypred[i]  = res[0][0]
          
            
        return Ypred
    
                
        