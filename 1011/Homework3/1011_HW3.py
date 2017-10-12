#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:26:32 2017

@author: sitibanc
"""
import math
import numpy as np
from sklearn import datasets

def entropy(p1, n1):        # postive, negative
    if p1 == 0 and n1 == 0:
        return 1
    value = 0
    pp = p1 / (p1 + n1)
    pn = n1 / (p1 + n1)
    if pp > 0:
        value -= pp * math.log2(pp)
    if pn > 0:
        value -= pn * math.log2(pn)
    return value

def infoGain(p1, n1, p2, n2):
    total = p1 + n1 + p2 + n2
    s1 = p1 + n1
    s2 = p2 + n2
    return entropy(p1 + p2, n1 + n2) - s1 / total * entropy(p1, n1) - s2 / total * entropy(p2, n2)

def buildDT(feature, target, positive, negative):
    ### node structure (dictionary)
    #   node.leaf = 0/1
    #   node.selectf = feature index
    #   node.threshold = some value (regards feature value)
    #   node.child = index of childs (leaft, right)
    ###
    # root node
    node = dict()
    node['data'] = range(len(target))
    
    ### tree structure (list)
    tree = []
    tree.append(node)
    ###
    
    i = 0
    while i < len(tree):
        idx = tree[i]['data']
        # data中的值是否相同
        if sum(target[idx] == negative) == len(idx):   #全負
            tree[i]['leaf'] = 1  # is leaf node
            tree[i]['decision'] = negative
        elif sum(target[idx] == positive) == len(idx):  #全正
            tree[i]['leaf'] = 1
            tree[i]['decision'] = positive
        # 試圖找出最好的切分方法
        else:
            bestIG = 0
            # 從該node(tree[j])中取出集合，決定threshold
            for j in range(feature.shape[1]):       # feature.shape回傳(rows長度, columns長度)的tuple
                pool = list(set(feature[idx, j]))   #以集合觀念處理去掉重複的值
                for k in range(len(pool) - 1):
                    threshold = (pool[k] + pool[k + 1]) / 2
                    G1 = []     #左子樹
                    G2 = []     #右子樹
                    for t in idx:
                        if feature[t, j] <= threshold:
                            G1.append(t)
                        else:
                            G2.append(t)
                    # Calculate infoGain
                    thisIG = infoGain(sum(target[G1] == positive), sum(target[G1] == negative), sum(target[G2] == positive), sum(target[G2] == negative))
                    # Update bestIG
                    if thisIG > bestIG:
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestThreshold = threshold
                        bestf = j
            if bestIG > 0:
                tree[i]['leaf'] = 0
                tree[i]['selectf'] = bestf
                tree[i]['threshold'] = bestThreshold
                tree[i]['child'] = [len(tree),len(tree) + 1]
                # 先放左子樹
                node = dict()
                node['data'] = bestG1
                tree.append(node)
                # 後放右子樹
                node = dict()
                node['data'] = bestG2
                tree.append(node)
            # 沒有更好的切分方法
            else:
                tree[i]['leaf'] = 1
                # 預測結果從多數決
                if sum(target[idx] == positive) > sum(target[idx] == negative):
                    tree[i]['decision'] = positive
                else:
                    tree[i]['decision'] = negative
        i += 1
        return tree

def testDT(tree, test_feature, test_target):
    now = 0
    while tree[now]['leaf'] == 0:
        bestf = tree[now]['selectf']
        threshold = tree[now]['threshold']
        # 屬於左子樹
        if test_feature[bestf] <= threshold:
            now = tree[now]['child'][0]
        # 屬於右子樹
        else:
            now = tree[now]['child'][1]
    if tree[now]['descion'] == test_target:
        return True
    else:
        return False
    
def predictDT(tree, test_feature):
    now = 0
    while tree[now]['leaf'] == 0:
        bestf = tree[now]['selectf']
        threshold = tree[now]['threshold']
        # 屬於左子樹
        if test_feature[bestf] <= threshold:
            now = tree[now]['child'][0]
        # 屬於右子樹
        else:
            now = tree[now]['child'][1]
    return tree[now]['descion']

### Main ###
# Load Data
iris = datasets.load_iris()
# Separate data & target according to its target value
data0 = iris.data[:50]
data1 = iris.data[50:100]
data2 = iris.data[100:]
targets0 = iris.target[:50]
targets1 = iris.target[50:100]
targets2 = iris.target[100:]

prediction = [0] * 150
error = 0

# Leave-one-out
for i in range(len(iris.data)):
    # Initial vote
    vote = [0, 0, 0]
    # Separate Feature & Target
    feature0 = data0
    feature1 = data1
    feature2 = data2
    target0 = targets0
    target1 = targets1
    target2 = targets2
    # Remove[i] data (Leave-one-out)
    if i < 50:
        feature0 = np.delete(feature0, i, 0)
        target0 = np.delete(target0, i, 0)
    elif i < 100:
        feature1 = np.delete(feature1, i % 50, 0)
        target1 = np.delete(target1, i % 50, 0)
    else:
        feature2 = np.delete(feature2, i % 50, 0)
        target2 = np.delete(target2, i % 50, 0)
    # Stack arrays in sequence vertically (row wise).
    tree01 = buildDT(np.vstack((feature0, feature1)), np.append(target0, target1), 0, 1)
    tree02 = buildDT(np.vstack((feature0, feature2)), np.append(target0, target2), 0, 2)
    tree12 = buildDT(np.vstack((feature1, feature2)), np.append(target1, target2), 1, 2)
    vote[predictDT(tree01, iris.data[i])] += 1
    vote[predictDT(tree02, iris.data[i])] += 1
    vote[predictDT(tree12, iris.data[i])] += 1
    # 檢查是否同票
    if max(vote) == 1:
        prediction[i] = 0
    else:
        for j in range(3):
            if vote[j] > 1:
                max_idx = j
        prediction[i] = j
# Calculate Error Rate
for i in range(len(iris.target)):
    if iris.target[i] != prediction[i]:
        error += 1
print('Error Rate:', error / len(iris.target))