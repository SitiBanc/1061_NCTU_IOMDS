# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:45:56 2017

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import math
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

fn = 0
ftable = []
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])

#sample is N-by-361 matrix
# return a vector with N feature values
def fe(sample,ftable,c):
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0):
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
    if(ftype==1):
        output = -np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)
    if(ftype==2):
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)+np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis=1)
    if(ftype==3):
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)-np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)
    return output

trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))

for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    
pw = np.ones((trpn,1))/trpn/2
nw = np.ones((trnn,1))/trnn/2

def WC(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max())
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
    polarity = 1
    if(error>0.5):
        error = 1-error
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10):
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            error = 1-error
            polarity = 0
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

SC = []
for t in range(10):
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0])
    best_feature = 0
    for i in range(1,fn):
        error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(error<best_error):
            best_feature = i
            best_error = error
            best_theta = theta
            best_polarity = polarity
    beta = best_error/(1-best_error)
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha])
    if(best_polarity==1):
        pw[trpf[:,best_feature]>=best_theta] = pw[trpf[:,best_feature]>=best_theta]*beta
        nw[trnf[:,best_feature]<best_theta] = nw[trnf[:,best_feature]<best_theta]*beta
    else:
        pw[trpf[:,best_feature]<best_theta] = pw[trpf[:,best_feature]<best_theta]*beta
        nw[trnf[:,best_feature]>=best_theta] = nw[trnf[:,best_feature]>=best_theta]*beta
        
trps = np.zeros((trpn,1))
trns = np.zeros((trnn,1))
alpha_sum = 0
for i in range(10):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum = alpha_sum + alpha
    if(polarity==1):
        trps[trpf[:,feature]>=theta] = trps[trpf[:,feature]>=theta]+alpha
        trns[trnf[:,feature]>=theta] = trns[trnf[:,feature]>=theta]+alpha
    else:
        trps[trpf[:,feature]<theta] = trps[trpf[:,feature]<theta]+alpha
        trns[trnf[:,feature]<theta] = trns[trnf[:,feature]<theta]+alpha

trps = trps/alpha_sum
trns = trns/alpha_sum