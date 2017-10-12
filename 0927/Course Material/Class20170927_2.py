# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:02:04 2017

@author: USER
"""
import math
from scipy.stats import norm

def blscall(S,L,T,r,sigma):
    d1 = (math.log(S/L)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1- sigma*math.sqrt(T)
    return S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)

def Bisection2(left,right,S,L,T,r,call,error):
    center = (left+right)/2
    if((right-left)/2<error):
        return center
    if((blscall(S,L,T,r,left)-call)*(blscall(S,L,T,r,center)-call)<0):
        return Bisection2(left,center,S,L,T,r,call,error)
    else:
        return Bisection2(center,right,S,L,T,r,call,error)
    
def Newton2(initsigma,S,L,T,r,call,iteration):
    sigma = initsigma
    for i in range(iteration):
        fx = blscall(S,L,T,r,sigma)-call
        fx2 = (blscall(S,L,T,r,sigma+0.00000001)-blscall(S,L,T,r,sigma-0.00000001))/0.00000002;
        sigma = sigma - fx/fx2
    return sigma

S = 10326.68
L = 10300.0
T = 21.0/365
r = 0.01065
sigma = 0.10508

print(blscall(S,L,T,r,sigma))
sigma = Bisection2(0.0000001,1,S,L,T,r,121.0,0.000000001)
print(sigma)
print(blscall(S,L,T,r,sigma))
print(Newton2(0.5,S,L,T,r,121.0,10))





