#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:13:01 2017

@author: sitibanc
"""
import sys
import math
import numpy as np
from PIL import Image
from numba import jit


def progress(count, total):
    '''
    count: int, current iteration
    total: int, total iteration times
    status: string, status message
    '''
    total_len = 50.0
    current_len = int(round(total_len * count / total))
    percents = round(count / total * 100.0, 1)
    bar = '=' * current_len + ' ' * (int(total_len) - current_len)
    if percents < 50:
        status = 'Patient!'
    elif percents < 85:
        status = 'Looking Good!'
    elif percents < 100:
        status = 'Almost There!'
    else:
        status = 'Finally Done!'
    sys.stdout.write('\r[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

@jit
def integral_image(image):
    # Add Padding to match the cords <- cords of index (0, 0) should be (1, 1)
    img = np.zeros((image.shape[0] + 1, image.shape[1] + 1))
    img[1:, 1:] = image.cumsum(0).cumsum(1)
    return img

@jit
def get_integral_images(images, shape):
    '''
    images: (N, 361) ndarray, flatten images
    shape: tuple(y, x), original shape of the image
    '''
    # Adjusting array's shape (add padding)
    new_width = (shape[0] + 1) * (shape[1] + 1)
    new_imgs = np.zeros((images.shape[0], new_width))
    for i in range(len(images)):
        img = integral_image(images[i, :].reshape(shape))
        new_imgs[i, :] = img.flatten()
    return new_imgs

@jit
def get_image_value(image, shape, top_left, bottom_right):
    '''
    image: (1, 400) ndarray, flatten integral image
    shape: tuple, original shape of image
    top_left: tuple(y, x), the index of the top left corner
    bottom_right: tuple(y, x), index of the bottom right corner
    '''
    # Reshape
    image = image.reshape(shape)
    A = image[top_left[0], top_left[1]]
    D = image[bottom_right[0], bottom_right[1]]
    B = image[top_left[0], bottom_right[1]]
    C = image[bottom_right[0], top_left[1]]
    return A - B - C + D

@jit
def build_ftable(img_shape, min_filter_shape, max_filter_shape):
    '''
    img_shape: tuple(y, x), image shape
    filter_shape: tuple(y, x), filter shape
    '''
    fn = 0
    ftable = []    # feature type, y, x, h, w
    height = img_shape[0]
    width = img_shape[1]
    # y, x: starting position; h, w: filter height & width
    for y in range(height):
        for x in range(width):
            for h in range(min_filter_shape[0], max_filter_shape[0] + 1):
                for w in range(min_filter_shape[1], max_filter_shape[1] + 1):
                    # check whether the bottom right position is still inside the image
                    if y + h <= height and x + w * 2 <= width:
                        fn += 1
                        ftable.append([0, y, x, h, w])
                    if y + h * 2 <= height and x + w <= width:
                        fn += 1
                        ftable.append([1, y, x, h, w])
                    if y + h  <= height and x + w * 3 <= width:
                        fn += 1
                        ftable.append([2, y, x, h, w])
                    if y + h * 2 <= height and x + w * 2 <= width:
                        fn += 1
                        ftable.append([3, y, x, h, w])
    return fn, ftable

@jit
def fe(sample,ftable,c):
    '''
    sample is N-by-361 matrix
    return a vector with N feature values
    '''
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

# Feature Extraction usding integral image
@jit
def integral_fe(sample, ftable, c, shape):
    '''
    sample: (N, 400) ndarray, (flatten integral image data)
    ftable: list, feature table(feature type, y, x, h, w)
    c: int, feature index (of ftable)
    shape: tuple(y, x), original shape of integral image
    '''
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    if ftype == 0:
        white = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y, x), bottom_right = (y + h, x + w))
        black = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y ,x + w), bottom_right =( y + h, x + w * 2))
    elif ftype == 1:
        white = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y + h, x), bottom_right = (y + h * 2, x + w))
        black = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y ,x), bottom_right =( y + h, x + w ))
    elif ftype == 2:
        white = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y, x), bottom_right = (y + h, x + w))
        white += np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y, x + w * 2), bottom_right = (y + h, x + w * 3))
        black = np.apply_along_axis(get_image_value, 1, sample, shape = shape , top_left = (y ,x + w), bottom_right =( y + h, x + w * 2))
    else:
        white = np.apply_along_axis(get_image_value, 1, sample, shape = shape, top_left = (y, x), bottom_right = (y + h, x + w))
        white += np.apply_along_axis(get_image_value, 1, sample, shape = shape, top_left = (y + h, x + w), bottom_right = (y + h * 2, x + w * 2))
        black = np.apply_along_axis(get_image_value, 1, sample, shape = shape, top_left = (y, x + w), bottom_right = (y + h, x + w * 2))
        black += np.apply_along_axis(get_image_value, 1, sample, shape = shape, top_left = (y + h, x), bottom_right = (y + h * 2, x + w))
    return white - black

# Weak Classifier
@jit
def WC(pw, nw, pf, nf):
    '''
    pw: (N, 1) ndarray, positive weights
    nw: (M, 1) ndrrray, negative weights
    pf: (N, 1) ndarray, positive features
    nf: (M, 1) ndarray, negative features
    '''
    maxf = max(pf.max(), nf.max())
    minf = min(pf.min(), nf.min())
    theta = (maxf - minf) / 10 + minf
    error = np.sum(pw[pf < theta]) + np.sum(nw[nf >= theta])
    polarity = 1
    if error > 0.5:
        error = 1 - error
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    # if better, update
    if error < min_error:
        min_theta = theta
        min_error = error
        min_polarity = polarity
    return min_error, min_theta, min_polarity

# =============================================================================
# Load Data
# =============================================================================
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
#testface = npzfile['arr_2']
#testnonface = npzfile['arr_3']

# =============================================================================
# Features Extraction
# =============================================================================
# data size
# tr: train/te: test; p:positive/n: negative; n: number/f: feature
trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
#tepn = testface.shape[0]
#tenn = testnonface.shape[0]

img_shape = (int(math.sqrt(trainface.shape[1])), int(math.sqrt(trainface.shape[1])))
integral_shape = ((img_shape[0] + 1, img_shape[1] + 1))

# Building ftable
print('Building ftable...')
# ftable stores every possible feature filter (starting) position & shape
# fn: total feature numbers
fn, ftable = build_ftable(img_shape,(2, 2), img_shape)

# Integral Image
#print('Calculating Integral Image...')
#trainface = get_integral_images(trainface, img_shape)
#trainnonface = get_integral_images(trainnonface, img_shape)
#testface = get_integral_images(testface, img_shape)
#testnonface = get_integral_images(testnonface, img_shape)

# feature numbers
trpf = np.zeros((trpn, fn))
trnf = np.zeros((trnn, fn))
#tepf = np.zeros((tepn, fn))
#tenf = np.zeros((tenn, fn))

# Get Features
print('Extracting Features...')
for c in range(fn):
    trpf[:, c] = fe(trainface, ftable, c)
    trnf[:, c] = fe(trainnonface, ftable, c)
#    trpf[:, c] = fe(trainface, ftable, c, integral_shape)
#    trnf[:, c] = fe(trainnonface, ftable, c, integral_shape)
#    tepf[:, c] = fe(testface, ftable, c, integral_shape)
#    tenf[:, c] = fe(testnonface, ftable, c, integral_shape)
    progress(c, fn)
# Load calculated data
#trpf = np.load('trpf.npy')
#trnf = np.load('trnf.npy')
#tepf = np.load('tepf.npy')
#tenf = np.load('tenf.npy')

# =============================================================================
# Adaboost Training
# =============================================================================
# weights
pw = np.ones((trpn, 1)) / trpn / 2
nw = np.ones((trnn, 1)) / trnn / 2

SC = []     # Strong Classifier
print('\nConstructing Strong Classifier...')
for t in range(10):
    weightsum = np.sum(pw) + np.sum(nw)
    pw = pw / weightsum
    nw = nw / weightsum
    # Use parameters 0f feature0 as the initial parameters
    best_error, best_theta, best_polarity = WC(pw, nw, trpf[:, 0], trnf[:, 0])
    best_feature = 0
    for i in range(1, fn):
        error, theta, polarity = WC(pw, nw, trpf[:, i], trnf[:, i])
        # if better, update
        if error < best_error:
            best_feature = i
            best_error = error
            best_theta = theta
            best_polarity = polarity
        progress(t * (fn - 1) + i ,(fn - 1) * 10)
    beta = best_error / (1 - best_error)
    alpha = math.log10(1 / beta)
    SC.append([best_feature, best_theta, best_polarity, alpha])
    # Adjusting Weights
    if best_polarity == 1:
        pw[trpf[:, best_feature] >= best_theta] *= beta
        nw[trnf[:, best_feature] < best_theta] *= beta
    else:
        pw[trpf[:, best_feature] < best_theta] *= beta
        nw[trnf[:, best_feature] >= best_theta] *= beta
# Calculate Scores
trps = np.zeros((trpn, 1))
trns = np.zeros((trnn, 1))
alpha_sum = 0
print('\nCalculating Scores...')
for i in range(10):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum += alpha
    if polarity == 1:
        trps[trpf[:, feature] >= theta] += alpha
        trns[trnf[:, feature] >= theta] += alpha
    else:
        trps[trpf[:, feature] < theta] += alpha
        trns[trnf[:, feature] < theta] += alpha
trps /= alpha_sum
trns /= alpha_sum

# =============================================================================
# Testing
# =============================================================================
# Read Image
I = Image.open('face.jpg')
data = np.asarray(I.convert('L'))
data_shape = data.shape
# Integral Image
test_integral = integral_image(data)
# Apply filter
windows = 0
filter_table = []
for i in range(3, 4):
    filter_shape = (19 * i, 19 * i)
    fn, ftable = build_ftable(data_shape, filter_shape, filter_shape)
    windows += fn
    filter_table.append(ftable)
# Apply SC

# Face Detection
threshold = 0.5
# Output result
I1 = Image.fromarray(data.astype('uint8'))