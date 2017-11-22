# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
def F1(t):
    return 0.063*(t**3) - 5.284*t*t + 4.887*t + 412 + np.random.normal(0,1)

def F2(t,A,B,C,D):
    return A*(t**B)+C*np.cos(D*t)+ np.random.normal(0,1,t.shape)

def E(b2,A2,A,B,C,D):
    return np.sum(abs(F2(A2,A,B,C,D)-b2))

n = 1000
b = np.zeros((n,1))
A3 = np.zeros((n,5))
for i in range(n):
    t = np.random.random()*100
    b[i] = F1(t)
    A3[i,0] = t**4
    A3[i,1] = t**3
    A3[i,2] = t**2
    A3[i,3] = t
    A3[i,4] = 1
x = np.linalg.lstsq(A3,b)[0]
print(x)

n = 1000
b2 = np.zeros((n,1))
A2 = np.random.random((n,1))*100
b2 = F2(A2,0.6,1.2,100,0.4)
print(E(b2,A2,0.6,1.2,100,0.4))
print(E(b2,A2,0.6,1.2,99,0.5))
    
pop = np.random.randint(0,2,(1000,40))
fit = np.zeros((1000,1))
for generation in range(10):
    print(generation)
    for i in range(1000):
        gene = pop[i,:]
        A = (np.sum(2**np.array(range(10))*gene[0:10])-511)/100
        B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        C = np.sum(2**np.array(range(10))*gene[20:30])-511
        D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
        fit[i]=E(b2,A2,A,B,C,D)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    for i in range(100,1000):
        fid = np.random.randint(0,100)
        mid = np.random.randint(0,100)
        while(mid==fid):
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(100):
        m = np.random.randint(0,1000)
        n = np.random.randint(0,40)
        if(pop[m,n]==0):
            pop[m,n]=1
        else:
            pop[m,n]=0


for i in range(1000):
    gene = pop[i,:]
    A = (np.sum(2**np.array(range(10))*gene[0:10])-511)/100
    B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
    C = np.sum(2**np.array(range(10))*gene[20:30])-511
    D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
    fit[i]=E(b2,A2,A,B,C,D)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]

gene = pop[0,:]
A = (np.sum(2**np.array(range(10))*gene[0:10])-511)/100
B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
C = np.sum(2**np.array(range(10))*gene[20:30])-511
D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100

    
    
    
    

