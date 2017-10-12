#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 22:06:40 2017

@author: sitibanc
"""
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

# 以9/27當日收盤的台指選為例

s = 10326.68    #目前股價
l = 10300.0     #執行價
t = 21.0 / 365  #距到期日
r = 0.01065     #無風險利率（台銀公告定存利率）
#sigma = 0.10503     # 波動率，未知，用猜der

# Black-Sholes定價模型
def blscall(s, l, t, r, sigma):
    d1 = math.log(s/l) + ( r + 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    #d2 = math.log(s/l) + ( r - 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm.cdf(d1) - l * math.exp(-r * t) * norm.cdf(d2)

# Bi-Section (guessing sigma)
def Bisection(left, right, s, l, t, r, call, iteration):
    centers = [0] * iteration
    for i in range(iteration):
        center = (left + right) / 2
        centers[i] = center
        if (blscall(s, l, t, r, left) - call) * (blscall(s, l, t, r, center) - call) < 0:
            right = center
        else:
            left = center
    return centers
    
# Newton-Raphson Method
def Newton(initSigma, s, l, t, r, call, iteration):
    sigmas = [0] * iteration
    sigma = initSigma
    for i in range(iteration):
        fx = blscall(s, l, t, r, sigma) - call
        fx2 = (blscall(s, l, t, r, sigma + 0.00000001) - blscall(s, l, t, r, sigma - 0.00000001 )) / 0.00000002
        sigma = sigma - fx / fx2
        sigmas[i] = sigma
    return sigmas

#HW2-2-1
print('HW2-2-1: ')
answer1 = Bisection(0.0000001, 1 , s, l, t, r, 121.0, 20)
plt.plot(answer1)
answer2 = Newton(0.5, s, l, t, r, 121.0, 20)
plt.plot(answer2)
plt.show()

#HW2-2-2
print('HW2-2-2: ')
ls = [10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10500.0, 10600.0, 10700.0]
calls = [354.0, 267.0, 187.0, 121.0, 69.0, 34.0, 14.5, 5.9]
answer3 = [0] * 8
for i in range(8):
    tmp = Newton(0.5, s, ls[i], t, r, calls[i], 20)
    answer3[i] = tmp[19]
plt.plot(answer3)
plt.show()