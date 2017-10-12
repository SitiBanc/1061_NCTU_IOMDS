#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:51:56 2017

@author: sitibanc
"""
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Black-Sholes定價模型
def blscall(s, l, t, r, sigma):
    d1 = math.log(s/l) + ( r + 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm.cdf(d1) - l * math.exp(-r * t) * norm.cdf(d2)

# Monte-Carlo Method
def genMC():
    n = 100
    dt = t / n
    p = np.zeros([10000, n+1])
    
    for i in range(10000):
        p[i,0] = s
        for j in range(n):
            p[i,j+1] = p[i,j] * math.exp((r - 0.5 * sigma ** 2) * dt + np.random.normal(0, 1, 1) * sigma * math.sqrt(dt))
    return p

# 以Mote-Carlo算出的結果回推目前call option的合理價格
def getMCPrice(n, p):
    c = 0
    for i in range(n):
        if p[i,100] > l:
            c += (p[i,100] - l) / i
    return c * math.exp(-r * t)

s = 50.0
l = 40.0
t = 2.0
r = 0.08
sigma = 0.2

bls = blscall(s, l, t, r, sigma)
p = genMC()
# HW2-1-1
print("HW2-1-1: ")
plt.plot(p[:200].T)

# HW2-1-2
# Get MCPrice list
dif = [0] * 100
j = 0
# Another approach
# 直接在getMCPrice()中處理
# 處理i，當i % 100 == 99時（第n百筆），將累加的c拿出來平均i + 1，折現後存到(i + 1) / 100 - 1
for i in range(100, 10001, 100):
    dif[j] = getMCPrice(i, p) - bls
    j += 1
plt.plot(dif)
    