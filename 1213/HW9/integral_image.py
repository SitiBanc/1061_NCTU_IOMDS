#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:41:23 2017

@author: sitibanc
"""
import numpy as np


def integral_image(image):
    return image.cumsum(0).cumsum(1)


def get_image_value(top_left, bottom_right, image):
    '''
    top_left: tuple, the index of the top left corner
    bottom_right: tuple, index of the bottom right corner
    image: ndarray, integral image
    '''
    A = image[top_left[1], top_left[0]]
    D = image[bottom_right[1], bottom_right[0]]
    B = image[top_left[1], bottom_right[0]]
    C = image[bottom_right[1], top_left[0]]
    return A + D - B - C


# main
test = np.ones((5, 5), dtype = np.int)
print(test)
test0 = integral_image(test)
print(test0)
print(get_image_value((0, 0), (2, 2), test0))