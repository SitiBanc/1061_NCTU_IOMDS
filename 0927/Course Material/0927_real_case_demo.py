#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:02:28 2017

@author: sitibanc
"""
import math
from scipy.stats import norm

# 以9/27當日收盤的台指選為例

s = 10326.68
l = 10300.0
t = 21.0 / 365
r = 0.01065
#sigma = 0.10503     # 波動率，未知，用猜der

# Black-Sholes定價模型

def blscall(s, l, t, r, sigma):
    d1 = math.log(s/l) + ( r + 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    #d2 = math.log(s/l) + ( r - 0.5 * sigma ** 2) * t / sigma * math.sqrt(t)
    d2 = d1 - sigma * math.sqrt(t)
    return s * norm.cdf(d1) - l * math.exp(-r * t) * norm.cdf(d2)

# Bi-Section (guess sigma)
def Bisection(left, right, s, l, t, r, call, error):      # error容錯率
    center = (left + right) / 2
    if (right - left) / 2 < error:
        return center
    if (blscall(s, l, t, r, left) - call) * (blscall(s, l, t, r, center) - call) < 0:
        return Bisection(left, center, s, l, t, r, call, error)
    else:
        return Bisection(center, right, s, l, t, r, call, error)
    
# Newton-Raphson Method
def Newton(initSigma, s, l, t, r, call, iteration):
    sigma = initSigma
    for i in range(iteration):
        fx = blscall(s, l, t, r, sigma) - call
        fx2 = (blscall(s, l, t, r, sigma + 0.00000001) - blscall(s, l, t, r, sigma - 0.00000001 )) / 0.00000002
        sigma = sigma - fx / fx2
        return sigma

sigma = Bisection(0.0000001, 1 , s, l, t, r, 121.0, 0.00001)
sigma2 = Newton(0.5, s, l, t, r, 121.0, 10)
print('Bisection Result (sigma): ', sigma)
print('Black-Scholes Call price: ', blscall(s, l, t, r, sigma))
print('Newton-Raphson Result(sigma): ', sigma2)
print('Black-Scholes Call price: ', blscall(s, l, t, r, sigma2))