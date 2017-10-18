#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:51:46 2017

@author: sitibanc
"""
import numpy as np
import matplotlib.pyplot as plt
import math, random


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

# Practice 1 :Generate simulated data
# G1: 標準差1,1; 平均4,4
G1 = np.random.normal(0, 1, (5000, 2))
G1 += 4
# G2: 標準差1, 3; 平均0, -3
G2 = np.random.normal(0, 1, (3000, 2))
G2[:, 1] = G2[:, 1] * 3 - 3

G = np.append(G1, G2, 0)

# G3: 標準差1,4; 平均-4, 6 ; 逆時針旋轉45度
G3 = np.random.normal(0, 1, (2000, 2))
G3[:, 1] = G3[:, 1] * 4
c45 = math.cos(-45 / 180 * math.pi)
s45 = math.sin(-45 / 180 * math.pi)
R = np.array([[c45, -s45], [s45, c45]])
G3 = G3.dot(R)
G3[:, 0] -= 4
G3[:, 1] += 6

G = np.append(G, G3, 0)
#plt.plot(G1[:,0], G1[:,1], 'r.')
#plt.plot(G2[:,0], G2[:,1], 'g.')
#plt.plot(G3[:,0], G3[:,1], 'b.')
#plt.show()

C, L, wicd = kmeans(G, 3, 1000)
print(wicd)
# Practice 2 : Draw K-Means Clustering Result
g1 = G[L == 0, :]
g2 = G[L == 1, :]
g3 = G[L == 2, :]

plt.plot(g1[:,0], g1[:,1], 'r.')
plt.plot(g2[:,0], g2[:,1], 'g.')
plt.plot(g3[:,0], g3[:,1], 'b.')
plt.plot(C[:, 0], C[:, 1], 'kx')
plt.show()

# Standard Score Normalization in one line
G0 =(G - np.tile(G.mean(0), (G.shape[0], 1))) / np.tile(G.std(0), (G.shape[0], 1))
# Standard Score Normalization using for-loop
for i in range(G.shape[1]):
    meanv = np.mean(G[:, i])
    stdv = np.std(G[:, i])
    G[:, i] = (G[:, i] - meanv) / stdv