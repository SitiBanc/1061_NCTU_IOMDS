#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:07:37 2017

@author: sitibanc
"""
import numpy as np
from scipy import signal
from PIL import Image


def conv2(I, M):
    # I: Image; M: Mask
    IH, IW = I.shape
    MH, MW = M.shape
    out = np.zeros((IH - MH + 1, IW - MW + 1))    # 無法完整讓Mask覆蓋的部份捨棄：原圖的長寬 - Mask的長寬 + 1
    # 輸出圖的長寬
    for h in range(IH - MH + 1):
        for w in range(IW - MW + 1):
            # Mask的長寬
            for y in range(MH):
                for x in range(MW):
                    out[h, w] = out[h, w] + I[h + y, w + x] * M[y, x]   # Mask起始位置固定在左上角(h + y, w + x)
    return out


# 讀圖
I = Image.open('sample2.jpg')
I.show()
data = np.asarray(I)

# =============================================================================
# # 產生負片效果
# data2 = 255 - data
# I2 = Image.fromarray(data2, 'RGB')
# I2.show()
# 
# # 分色
# data3 = np.zeros((917, 516, 3)).astype('uint8')
# data3[:, :, 1] = data[:, :, 1]                      # 分出綠色
# I3 = Image.fromarray(data3, 'RGB')
# I3.show()
# =============================================================================

# MeanBlur
M = np.ones((10, 10)) / 100    # re-weighted
# =============================================================================
# Usee our function
# R = data[:, :, 0]
# G = data[:, :, 1]
# B = data[:, :, 2]
# R2 = conv2(R, M)
# G2 = conv2(G, M)
# B2 = conv2(B, M)
# data4 = data.copy()
# data4[:, :, 0] = R2.astype('uint8')
# data4[:, :, 1] = G2.astype('uint8')
# data4[:, :, 2] = B2.astype('uint8')
# I4 = Image.fromarray(data4, 'RGB')
# I4.show()
# =============================================================================

# Using SciPy
R = data[:, :, 0]
R2 = signal.convolve2d(R, M, mode = 'same', boundary = 'symm')
G = data[:, :, 1]
G2 = signal.convolve2d(G, M, mode = 'same', boundary = 'symm')
B = data[:, :, 2]
B2 = signal.convolve2d(B, M, mode = 'same', boundary = 'symm')

data4 = data.copy()
data4[:, :, 0] = R2.astype('uint8')
data4[:, :, 1] = G2.astype('uint8')
data4[:, :, 2] = B2.astype('uint8')
I4 = Image.fromarray(data4, 'RGB')
I4.show()