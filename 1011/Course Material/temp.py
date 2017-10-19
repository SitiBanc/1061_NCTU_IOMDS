#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 22:25:38 2017

@author: sitibanc
"""
import math
import numpy as np

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
    
data = np.loadtxt('PlayTennis.txt',usecols=range(5),dtype=int)
feature = data[:,0:4]
target = data[:,4]-1

def DT(feature, target):
    node = dict()
    node['data'] = range(len(target))
    Tree = [];
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            print(idx)
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            print(idx)
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = infoGain(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target(idx)==1)>sum(target(idx)==0)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=0
        t+=1
    return Tree

Tree = buildDT(feature, target, 1, 0)
#Tree = DT(feature, target)

for i in range(len(target)):
    test_feature = feature[i,:]
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
    print(target[i],Tree[now]['decision'])
    