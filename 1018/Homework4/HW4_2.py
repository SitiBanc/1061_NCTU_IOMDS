#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:36:02 2017

@author: sitibanc
"""
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


def knn(feature, target, K):
    N = feature.shape[0]                # N筆資料
    L = np.zeros((N, 1))                # Label (data屬於哪一個class)
    dist = np.zeros((N, N))             # N筆資料與其他筆資料的距離
    vote = np.zeros((N, list(set(target))))
    
    # Calculate Distance
    for i in range(N):
        dist[:, i] = np.sum((np.tile(feature[i, :], (N, 1)) - feature) ** 2, 1)
    # Get Minmun Distance & Vote For The Label
    for i in range(N):
        
    return L


# Practice 5 : K-NN Classifier & Confusion Matrix
data = datasets.load_iris()
feature = data.data
target = data.target
L = knn(feature, target, 1)