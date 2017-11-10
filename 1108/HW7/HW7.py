#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:25:18 2017

@author: sitibanc
"""
import numpy as np
from scipy import signal
from PIL import Image


def gen2DGaussian(stdv, mean, h, w):
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x ** 2 + y ** 2)
    sigma, mu = stdv, mean
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g

def applyMask(M, I_array):
    R = I_array[:, :, 0]
    R2 = signal.convolve2d(R, M, mode = 'same', boundary = 'symm')
    G = I_array[:, :, 1]
    G2 = signal.convolve2d(G, M, mode = 'same', boundary = 'symm')
    B = I_array[:, :, 2]
    B2 = signal.convolve2d(B, M, mode = 'same', boundary = 'symm')
    
    data = I_array.copy()
    data[:, :, 0] = R2.astype('uint8')
    data[:, :, 1] = G2.astype('uint8')
    data[:, :, 2] = B2.astype('uint8')
    return data

# 讀圖
I = Image.open('sample.jpg')
data = np.asarray(I)

# =============================================================================
# HW7-1: Gaussian Blur
# =============================================================================
# Generate 2D Gaussian Array
M1 = gen2DGaussian(1.0, 0.0, 10, 10)
M1 = M1 / M1.sum()
# Apply Mask
masked1 = applyMask(M1, data)
I1 = Image.fromarray(masked1, 'RGB')
I1.show()

# =============================================================================
# HW7-2: Motion Blur
# =============================================================================
M2 = np.ones((20, 1))
M2 = M2 / M2.sum()
# Apply Mask
masked2 = applyMask(M2, data)
I2 = Image.fromarray(masked2, 'RGB')
I2.show()

# =============================================================================
# HW7-3: Sharp Filter（銳化）    <-- 兩個標準差不同的Gaussian相減
# =============================================================================
# Generate Mask
#sig1 = gen2DGaussian(1.0, 0.0, 3, 3)
#sig2 = gen2DGaussian(2.0, 0.0, 3, 3)
#M3 = sig1 - sig2
#M3 = M3 / M3.sum()
# Another Mask
M3 = np.array([[-1, -1, -1], [-1, 16, -1], [-1, -1, -1]])
M3 = M3 / 8
# Apply Mask
masked3 = applyMask(M3, data)
I3 = Image.fromarray(masked3, 'RGB')
I3.show()

# =============================================================================
# HW7-4: Sobel Filter（邊界強化、類似素描風格）
# =============================================================================
# Gray-scale image
I0 = I.convert('L')
data0 = np.asarray(I0)
# Generate Mask
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# Apply Mask
Ix = signal.convolve2d(data0, sobel_x, mode = 'same', boundary = 'symm')
Iy = signal.convolve2d(data0, sobel_y, mode = 'same', boundary = 'symm')
masked4 = Ix ** 2 + Iy ** 2
# Adjust Color
tmp = masked4.flatten()
tmp[::-1].sort()    # sorting in descending order
n = 0.2
idx = int(len(tmp) * n)
for h in range(masked4.shape[0]):
    for w in range(masked4.shape[1]):
         if masked4[h, w] >= tmp[idx]:
             masked4[h, w] = 255
         else:
             masked4[h, w] = 0
I4 = Image.fromarray(masked4, 'L')
I4.show()