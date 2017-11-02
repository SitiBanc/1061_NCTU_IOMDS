# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:41:20 2017

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from PIL import Image
import numpy as np
import math
import random


def loadIMG(file_path):
    # Load File
    os.chdir(file_path)                                 # Change directory
    filelist = os.listdir()                             # 檔案位置list
    x = np.zeros((len(filelist), 19 * 19))              # 準備檔案數 * 361 pixels
    # Read Image
    for i in range(len(filelist)):
        IMG = Image.open(filelist[i])
        x[i, :] = np.array(IMG.getdata())               # 將IMG中19*19的資料拉成1維陣列
    return x


def BPNNtrain(pf,nf,hn,lr,iteration):
    pn = pf.shape[0]
    nn = nf.shape[0]
    fn = pf.shape[1]
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0)
    model = dict()
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1)
            ho = ins.dot(WI)
            for j in range(hn):
                ho[j] = 1/(1+math.exp(-ho[j]))
            hs = np.append(ho,1)
            out = hs.dot(WO)
            out = 1/(1+math.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out)
            dh = ho*(1-ho)*WO[0:hn,0]*dk
            WO[:,0] = WO[:,0] + lr*dk*hs
            for j in range(hn):
                WI[:,j] = WI[:,j] + lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO                                
    return model

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    hn = WI.shape[1]
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI)
        for j in range(hn):
            ho[j] = 1/(1+math.exp(-ho[j]))
        hs = np.append(ho,1)
        out[i] = hs.dot(WO)
        out[i] = 1/(1+math.exp(-out[i]))
    return out

# Load Data
trainface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/face')
trainnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/non-face')
testface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/face')
testnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/non-face')

network = BPNNtrain(trainface/255,trainnonface/255,20,0.01,10)
pscore = BPNNtest(trainface/255,network)
nscore = BPNNtest(trainnonface/255,network)