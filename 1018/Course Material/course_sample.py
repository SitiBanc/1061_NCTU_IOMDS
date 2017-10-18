# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))
    L = np.zeros((N,1))
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)
        L1 = np.argmin(dist,1)
        if(iter>0 and np.array_equal(L,L1)):
            break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter += 1
    wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    return C,L,wicd


    
G1 = np.random.normal(0,1,(5000,2))
G1 = G1+4
G2 = np.random.normal(0,1,(3000,2))
G2[:,1] = G2[:,1]*3 - 3
G = np.append(G1,G2,axis=0)
G3 = np.random.normal(0,1,(2000,2))
G3[:,1] = G3[:,1]*4
c45 = math.cos(-45/180*math.pi)
s45 = math.sin(-45/180*math.pi)
R = np.array([[c45,-s45],[s45,c45]])
G3 = G3.dot(R)
G3[:,0] = G3[:,0]-4
G3[:,1] = G3[:,1]+6
G = np.append(G,G3,axis=0)
plt.plot(G[:,0],G[:,1],'.')
C,L,wicd = kmeans(G,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
print(wicd)

G = (G-np.tile(G.mean(0),(G.shape[0],1)))/np.tile(G.std(0),(G.shape[0],1))
for i in range(G.shape[1]):
    meanv = np.mean(G[:,i])
    stdv = np.std(G[:,i])
    G[:,i] = (G[:,i]-meanv)/stdv




