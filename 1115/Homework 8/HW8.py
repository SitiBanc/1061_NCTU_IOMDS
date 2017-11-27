#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:07:44 2017

@author: sitibanc
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Linear Regression
def F1(t):
    return 0.063 * (t ** 3) - 5.284 * t * t + 4.887 * t + 412 + np.random.normal(0, 1)


# Non-Linear
def F2(t, A, B, C, D):
    return A * (t ** B) + C * np.cos(D * t) + np.random.normal(0, 1, t.shape)


# Energy Function
def E(targets, t, A, B, C, D):
    return np.sum(abs(F2(t, A, B, C, D) - targets))


# =============================================================================
# Experiment 1: Fix A, B, C, try different D
# =============================================================================
n = 1000
# True answers (targets) of random t
targets = np.zeros((n, 1))
# Random t (data)
t = np.random.random((n, 1)) * 100
# Calculate true ans
for i in range(n):
    targets[i] = F2(t[i], 0.6, 1.2, 100, 0.4)
# Try different Ds
Ds = np.array(range(-511, 512)) / 100
# D & correspond Energy
exp1 = np.zeros((len(Ds), 2))
for i in range(len(Ds)):
    exp1[i, 0] = Ds[i]
    exp1[i, 1] = E(targets, t, 0.6, 1.2, 100, Ds[i])
# Plot
plt.plot(exp1[:, 0], exp1[:, 1])
plt.show()

# =============================================================================
# Experiment 2: Fix B, D, try different A, C
# =============================================================================
n = 1000
# True Ans
targets2 = np.zeros((n, 1))
# Random t
t2 = np.random.random((n,1))*100
for i in range(n):
    targets2[i] = F2(t2[i], 0.6, 1.2, 100, 0.4)
As = np.array(range(-511, 512)) / 100
Cs = np.array(range(-511, 512)) / 100
X = np.zeros((len(As), len(Cs)))
Y = np.zeros((len(As), len(Cs)))
Z = np.zeros((len(As), len(Cs)))
for j in range(len(As)):
    for k in range(len(Cs)):
        X[k, j] = As[j]
        Y[k, j] = Cs[k]
        Z[k, j] = E(targets2, t2, As[j], 1.2, Cs[k], 0.4)
# Plot
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_zlim(60000, 640000)
surf = ax.plot_surface(X, Y, Z, cmap = cm.jet, rstride = 1, cstride = 1, linewidth = 0)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()
# =============================================================================
# Experiment 3: LPPL
# =============================================================================
# Read Data
data = np.loadtxt('Bubble.txt')
# Genetic Algorithm
# Genes of Initial Population
pop = np.random.randint(0, 2, (10000, 40))
fit = np.zeros((10000, 1))

for generation in range(100):
    print('Generation:', generation)
    # Reproduction
    for i in range(10000):
        # Calculate Fitness using Energy Function
        gene = pop[i, :]
        # binary to decimal
        tc = np.sum(2 ** np.array(range(10)) * gene[:10] - 511) / 100
        beta = np.sum(2 ** np.array(range(10)) * gene[10:20] - 511) / 100
        omega = np.sum(2 ** np.array(range(10)) * gene[20:30] - 511)
        phi = np.sum(2 ** np.array(range(10)) * gene[30:] - 511) / 100
        fit[i] = E(targets, t, tc, beta, omega, phi)
    # Using Tourment Select to decide survival
    sortf = np.argsort(fit[:, 0])
    pop = pop[sortf, :]
    # Crossover using mask (0: mother; 1: father)
    for i in range(100, 10000):
        # fid: father id; mid: mother id
        fid = np.random.randint(0, 100)
        mid = np.random.randint(0, 100)
        while mid == fid:
            mid = np.random.randint(0, 100)
        mask = np.random.randint(0, 2, (1, 40))
        child = pop[mid, :]
        father = pop[fid, :]
        child[mask[0, :] == 1] = father[mask[0, :] == 1]
        pop[i, :] = child
    # Mutation
    for i in range(100):
        # the mutated ones
        m = np.random.randint(0, 10000)
        # the mutated genes
        n = np.random.randint(0, 40)
        if pop[m, n] == 1:
            pop[m, n] = 0
        else:
            pop[m, n] = 1
# Linear Regression