#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:07:02 2017

@author: sitibanc
"""
import os, math, random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def BPNNtrain(pf, nf, hn, lr, iteration):
    # postive feature, negative featue, hidden nodes number, learning rate, learning times
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
            out = 1 / (1 + math.exp(-out))
            # Gradient descent
            dk = out * (1 - out) * (target[s[i]] - out) # delta value of output node
            dh = ho * (1 - ho) * WO[0:hn, 0] * dk       # delta value of hidden nodes
            # Update weights
            WO[:, 0] = WO[:, 0] + lr * dk * hs
            for j in range(hn):
                WI[:, j] = WI[:, j] + lr * dh[j] * ins
    model['WI'] = WI
    model['WO'] = WO
    return model


def BPNNtest(feature, model):
    sn = feature.shape[0]                               # sample number
    WI = model['WI']                                    # input to hidden layer weights
    WO = model['WO']                                    # hidden to output layer weights
    hn = WI.shape[1]                                    # hidden nodes number
    out = np.zeros((sn, 1))                             # model predict value
    for i in range(sn):
        ins = np.append(feature[i, :], 1)               # adding constant
        ho = ins.dot(WI)                                # multiply input-to-hidden weights
        for j in range(hn):
            ho[j] = 1 / (1 + math.exp(-ho[j]))          # apply sigmoid function
        hs = np.append(ho, 1)                           # adding constant
        out[i] = hs.dot(WO)                             # multiply hidden-to-output weights
        out[i] = 1 / (1 + math.exp(-out[i]))            # apply sigmoid function
    return out


def calROC(network, p, n, hn, lr, t):
    pscore = BPNNtest(p, network)
    nscore = BPNNtest(n, network)
    # Calculate TPR & FNR with different thresholds
    ROC = np.zeros((100, 2))
    for t in range(100):
        # Calculate True Postive Rate & False Negative Rate
        threshold = (t + 1) / 100
        for i in range(len(pscore)):
            if pscore[i, 0] >= threshold:
                ROC[t, 0] += 1 / len(pscore)                # True Positive / Actual Positive
        for i in range(len(nscore)):
            if nscore[i, 0] < threshold:
                ROC[t, 1] += 1 / len(nscore)                # False Negative / Actual Negative
    return ROC


# Load Data
trainface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/face') / 255
trainnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/train/non-face') / 255
testface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/face') / 255
testnonface = loadIMG('/home/sitibanc/1061_NCTU_IOMDS/1101/Course Material/CBCL/test/non-face') / 255

# Test hidden nodes number
test_hn = [20, 30, 40]
for i in range(len(test_hn)):
    network = BPNNtrain(trainface, trainnonface, 20, 0.01, 10)
    ROC = calROC(network, trainface, trainnonface, 20, 0.01, 10)
    plt.plot(ROC[:, 0], ROC[:, 1])
    ROC = calROC(network, testface, testnonface, 20, 0.01, 10)
    plt.plot(ROC[:, 0], ROC[:, 1])
    print('Hidden Nodes Number:', hn[i], '\nLearning Rate:', lr, '\nIteration Times:', it)
    plt.show()