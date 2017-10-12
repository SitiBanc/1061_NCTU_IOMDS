#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:05:27 2017

@author: sitibanc
"""
import math
from scipy.stats import norm
import numpy as np

# Black-Sholes定價模型

def blscall(s, l, t, r, sigma):
    d1 = math.log(s/l) + ( r + 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    #d2 = math.log(s/l) + ( r - 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm.cdf(d1) - l * math.exp(-r * t) * norm.cdf(d2)

s = 50.0
l = 40.0
t = 2.0
r = 0.08
sigma = 0.2

print('Black-Scholes Call price: ', blscall(s, l, t, r, sigma))
print('Parity: ', blscall(s, l, t, r, sigma) + l * math.exp(-r * t) - s)

# 差分代替偏微分

d1 = math.log(s/l) + ( r + 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
print('偏微分：', norm.cdf(d1))
print('差分：', (blscall(s + 0.01, l, t, r, sigma) - blscall(s - 0.01, l, t, r, sigma)) / 0.02)


# Monte-Carlo Method

n = 100
dt = t / n

p = np.zeros([10000, n+1])
for i in range(10000):
    p[i,0] = s
    for j in range(n):
        p[i,j+1] = p[i,j] * math.exp((r - 0.5 * sigma ** 2) * dt + np.random.normal(0, 1, 1) * sigma * math.sqrt(dt))

# 以Mote-Carlo算出的結果回推目前call option的合理價格
c = 0
for i in range(10000):
    if p[i, 100] > l:
        c += (p[i, 100] - l) / 10000
print('Monte-Carlo: ', c * math.exp(-r * t))