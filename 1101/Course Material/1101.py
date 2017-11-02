#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:19:49 2017

@author: sitibanc
"""
import os, math, random
from PIL import Image
import numpy as np


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


def BPNN(pf, nf, hn, lr, iteration):
    # postive feature, negative featuew, hidden nodes number, learning rate, learning times
    pn = pf.shape[0]
    nn = nf.shape[0]
    fn = pf.shape[1]                                    # feature number
    model = dict()
    # Randomly assign initial weights
    WI = np.random.normal(0, 1, (fn + 1, hn))           # input to hidden layer weights (plus 1 means constant aka w0)平移sigmoid fumction
    WO = np.random.normal(0, 1, (hn + 1, 1))            # hidden to output layer weights
    
    feature = np.append(pf, nf, axis = 0)
    target = np.append(np.ones((pn, 1)), np.zeros((nn, 1)), axis = 0)
    for t in range(iteration):
        s = random.sample(range(pn + nn), pn + nn)      # shuffle training data (mix positive and negative data)
        for i in range(pn + nn):
            ins = np.append(feature[s[i], :], 1)        # input signal (adding constant) <-- actual output
            ho = ins.dot(WI)                            # hidden layer output (matrix multiplication)
            for j in range(hn):
                ho[j] = 1 / (1 + math.exp(-ho[j]))      # sigmoid function (constraint value to be within 0~1)
            hs = np.append(ho, 1)                       # hidden layer signal (adding constant) <-- actual output
            out = hs.dot(WO)                            # matrix multiplication (multiply weights)
            # Gradient descent
            dk = out * (1 - out)(target[s[i]] - out)    # delta value of output node
            dh = np.zeros((hn, 1))                      # delta value of hidden nodes
            for j in range(hn):
                dh[j] = ho[j] * (1 - ho[j]) * WO[j] *dk
            # Update weights
            WO = WO + lr * dk * hs
            for j in range(hn):
                WI[:, j] = WI[:, j] + lr * dh[j] * ins
    model['WI'] = WI
    model['WO'] = WO
    return model


def BPNNtest(feature, model):
    sn = feature.shape[0]                               # sample number
    WI = model['WI']
    WO = model['WO']
    ins = np.append(feature[i, :], 1)
    ho = ins.dot(WI)
    for i in range(sn):
        ins = np.append(feature[i, :], 1)

# Load Data
trainface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/face')
trainnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/non-face')
testface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/face')
testnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/non-face')

network = BPNN(trainface, trainnonface, 20, 0.2, 10)