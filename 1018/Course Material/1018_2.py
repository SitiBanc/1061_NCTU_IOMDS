#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:53:35 2017

@author: sitibanc
"""
import numpy as np
import random
from sklearn import datasets


def kmeans(sample, K, max_iter):
    N = sample.shape[0]                 # N筆資料
    D = sample.shape[1]                 # 每筆資料有N維
    C = np.zeros((K, D))                # K個中心點
    L = np.zeros((N, 1))                # Label (data屬於哪一個cluster)
    L1 = np.zeros((N, 1))               # 重新分群計算出的label
    dist = np.zeros((N, K))
    
    # Random select center
    idx = random.sample(range(N), K)
    C = sample[idx, :]
    iteration = 0
    
    while iteration <= max_iter:
        for i in range(K):
            #以整個矩陣做運算，求與C（中心點）距離
            # np.tile() --> 垂直Repeat C[i, :] N次，水平repeat 1次
            dist[:, i] = np.sum((sample - np.tile(C[i, :], (N, 1))) ** 2 , 1)
            # 取距離最短者的input為其label
            L1 = np.argmin(dist, 1)
            # 若分群後各群成員不再改變（已分完，所屬cluster已定），則跳出迴圈
            if iteration > 0 and np.array_equal(L, L1):
                break
            # Update Label L
            L = L1
            # 計算重新分群後的新center
            for i in range(K):
                # 取出屬於第i群者的index
                idx = np.nonzero(L == i)[0]    # np.nonzero()亦即True
                if len(idx) > 0:
                    C[i, :] = np.mean(sample[idx, :], 0)    # 沿垂直方向(0)計算平均  
        iteration += 1
        # Calcuate wicd（within cluster distance, 群內每筆資料與群中心的距離）
        wicd = np.sum(np.sqrt(np.sum((sample - C[L, :]) ** 2 , 1)))
    return C, L, wicd

data = datasets.load_iris()