#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:19:35 2017

@author: sitibanc
"""
import math

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