#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:49:00 2017

@author: sitibanc
"""
import numpy as np
from numpy import matlib


def PCATrain(D, R):
    '''
    D: ndarray, Data Matrix(with M samples * F features)
    R: float, Ratio(the lower bound of information preservation)
    '''
    M, F = D.shape
    meanv = np.mean(D, axis = 0)
    # Repeate Matrix(repeat meanv M times along first axis, 1 times along second axis)
    D2 = D - matlib.repmat(meanv, M, 1)
    C = np.dot(D2.T, D2)
    # eigenvalue, eigenvector
    EValue, Evector = np.linalg.eig(C)
    EV2 = np.cumsum(EValue) / np.sum(EValue)
    num = np.where(EV2 >= R)[0][0] + 1
    return meanv, Evector[:, range(num)]


def PCATest(D, meanv, W):
    M, F = D.shape
    D2 = D - matlib.repmat(meanv, M, 1)
    D3 = np.dot(D2, W)
    return D3


# Main
D = np.load('pca_data.npy')
# Train
meanv, W = PCATrain(D, 0.9)
D3 = PCATest(D, meanv, W)