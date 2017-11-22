#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:46:19 2017

@author: sitibanc
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

# Load Data
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

x = np.append(trainface, trainnonface, axis = 0)
y = np.append(np.ones((trainface.shape[0], 1)), np.zeros((trainnonface.shape[0], 1)))
x2 = np.append(testface, testnonface, axis = 0)
y2 = np.append(np.ones((testface.shape[0], 1)), np.zeros((testnonface.shape[0], 1)))

# Training Using Multi-Layer Perceptron
clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5, 8), random_state = 1)
clf.fit(x, y)

y_pred = clf.predict(x)
train_accuracy = np.sum(y == y_pred) / y.shape[0]
print('Training Accuracy:', train_accuracy * 100, '%')
y2_pred = clf.predict(x2)
test_accuracy = np.sum(y2 == y2_pred) / y2.shape[0]
print('Testing Accuracy:', test_accuracy * 100, '%')

# Standardizations


# Symmetric Face