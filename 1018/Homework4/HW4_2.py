#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:36:02 2017

@author: sitibanc
"""
from sklearn import datasets
import numpy as np

# K-NN Classifier
def knn(feature, target, K):
    N = feature.shape[0]                # N筆資料
    L = np.zeros((N, 1))                # Label (data屬於哪一個class)
    dist = np.zeros((N, N))             # N筆資料與其他筆資料的距離
    
    # Calculate Distance
    for i in range(N):
        dist[:, i] = np.sum((np.tile(feature[i, :], (N, 1)) - feature) ** 2, 1)
    # Get Minmun Distance
    for i in range(N):
        vote = np.zeros((1, len(set(target))))
        current = dist[:, i]
        count = 0
        # Vote For The Label
        while count < K:
            idx = np.argmin(current, 0)
            if idx != i:
                vote_idx = target[idx]
                vote[0, vote_idx] += 1
                count += 1
            current = np.delete(current, (idx), axis = 0)
        L[i, 0] = np.argmax(vote, 1)
        
    return L

# Building Confusion Matrix
def buildCF(L, T):
    t = len(set(T))
    cf = np.zeros((t, t))
    for i in range(len(T)):
        cf[int(L[i, 0]), T[i]] += 1
    return cf

# Practice 5 : K-NN Classifier & Confusion Matrix

# Load Data
data = datasets.load_iris()
feature = data.data
target = data.target

# Try K = 1~10
cfs = [0] * 10
for i in range(10):
    L = knn(feature, target, i + 1)
    cfs[i] = buildCF(L, target)