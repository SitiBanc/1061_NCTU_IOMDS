#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:07:44 2017

@author: sitibanc
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


# Linear Regression
def F1(t):
    return 0.063 * (t ** 3) - 5.284 * t * t + 4.887 * t + 412 + np.random.normal(0, 1)


# Non-Linear
def F2(t, A, B, C, D):
    return A * (t ** B) + C * np.cos(D * t) + np.random.normal(0, 1, t.shape)


# Energy Function
def E(targets, t, A, B, C, D):
    return np.sum(abs(F2(t, A, B, C, D) - targets))


# Energy Function for LPPL
def E2(ans, A, B, C, tc, t, beta, omega, phi):
    return np.sum(abs(LPPL(A, B, C, tc, t, beta, omega, phi) - ans))


# Log-Periodic Power Laws for bubble modeling
def LPPL(A, B, C, tc, t, beta, omega, phi):
    return A + (B * np.power(tc - t, beta)) * (1 + (C * np.cos(omega * np.log(tc - t) + phi)))


# Genes Decoding
def decodeGenes(gene):
    # binary to decimal
    # tc: 6 bits, range from 550 ~ 613
    tc = np.sum(2 ** np.array(range(6)) * gene[:6]) + 550
    # beta: 8 bits, range from 1/257 ~ 256/257
    beta = (np.sum(2 ** np.array(range(8)) * gene[6:14]) + 1) / 257
    # omega: 4 bits, range from 5 ~ 20
    omega = np.sum(2 ** np.array(range(4)) * gene[14:18]) + 5
    # phi: 6 bits, range from 0.01 ~ 63.02/2pi
    phi = (np.sum(2 ** np.array(range(6)) * gene[18:]) + 0.01) / (63.02 / (np.pi * 2))
    return tc, beta, omega, phi

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
Cs = np.array(range(-511, 512))
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
surf = ax.plot_surface(X, Y, Z, cmap = cm.jet, rstride = 1, cstride = 1, linewidth = 0)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.show()

# =============================================================================
# Experiment 3: LPPL
# =============================================================================
# Read Data
data = np.loadtxt('Bubble.txt')
price = data[:550, 1]
# Genetic Algorithm
# Genes of Initial Population
pop = np.random.randint(0, 2, (10000, 24))
fit = np.zeros((len(pop), 1))
# Parameters
# tc, beta, omega, phi, A, B, C
parameters = np.zeros((len(pop), 7))

for generation in range(50):
    print('---------- Generation', generation + 1, '----------')
    # Calculate Fitness using Energy Function
    print('Calculate Fitness...')
    for i in range(len(pop)):
        # Decode Genes
#        print('Genes Decoding...')
        parameters[i, 0], parameters[i, 1], parameters[i, 2], parameters[i, 3] = decodeGenes(pop[i, :])
        
#        print('Linear Regression...')
        # Linear Regression
        coefs = np.zeros((len(price), 3))
        # A's coefficient
        coefs[:, 0] += 1
        for t in range(len(price)):
            # B's coefficient
            coefs[t, 1] = np.power(parameters[i, 0] - t, parameters[i, 1])
            # BC's coefficient
            coefs[t, 2] = coefs[t, 1] * np.cos((parameters[i, 2] * np.log(parameters[i, 0] - t)) + parameters[i, 3])
        # Least Square
        x = np.linalg.lstsq(coefs, price)[0]
        parameters[i, 4] = x[0]
        parameters[i, 5] = x[1]
        parameters[i, 6] = x[2] / parameters[i, 5]
#        print('Calculate Fitness...')
        fit[i] = E2(price, parameters[i, 4], parameters[i, 5], parameters[i, 6], parameters[i, 0], np.array(range(len(price))), parameters[i, 1], parameters[i, 2], parameters[i, 3])
        
    # Elimination
    # Using Tourment Select to decide survival
    # Top 100 survived, others will be replaced during crossover
    # Sorting population according to its fitness
    print('Eliminating...')
    sortf = np.argsort(fit[:, 0])
    pop = pop[sortf, :]
    parameters = parameters[sortf, :]
    fit = fit[sortf]
    
    # Reproduction
    # Crossover using mask (0: mother; 1: father)
    # Top 100 survived, crossover till population is filled
    print('Reproducing...')
    for i in range(100, len(pop)):
        # fid: father id; mid: mother id
        fid = np.random.randint(0, 100)
        mid = np.random.randint(0, 100)
        while mid == fid:
            mid = np.random.randint(0, 100)
        mask = np.random.randint(0, 2, (1, pop.shape[1]))
        # Copy mother
        child = pop[mid, :]
        father = pop[fid, :]
        # Apply mask (if mask == 1, use father's genes)
        child[mask[0, :] == 1] = father[mask[0, :] == 1]
        pop[i, :] = child
        
    # Mutation
    # Randomly select 100 ones from the population that's gonna be mutated
    print('Mutating...')
    for i in range(100):
        # the mutated ones
        m = np.random.randint(0, pop.shape[0])
        # the mutated genes
        n = np.random.randint(0, pop.shape[1])
        # Switch genes
        if pop[m, n] == 1:
            pop[m, n] = 0
        else:
            pop[m, n] = 1

# Calculate Fitness
print('---------- Final Result ----------')
print('Calculate Final Fitness...')
for i in range(len(pop)):
        # Decode Genes6
        parameters[i, 0], parameters[i, 1], parameters[i, 2], parameters[i, 3] = decodeGenes(pop[i, :])
        # Linear Regression
        coefs = np.zeros((len(price), 3))
        # A's coefficient
        coefs[:, 0] += 1
        for t in range(len(price)):
            # B's coefficient
            coefs[t, 1] = np.power(parameters[i, 0] - t, parameters[i, 1])
            # BC's coefficient
            coefs[t, 2] = coefs[t, 1] * np.cos((parameters[i, 2] * np.log(parameters[i, 0] - t)) + parameters[i, 3])
        # Least Square
        x = np.linalg.lstsq(coefs, price)[0]
        parameters[i, 4] = x[0]
        parameters[i, 5] = x[1]
        parameters[i, 6] = x[2] / parameters[i, 5]
        fit[i] = E2(price, parameters[i, 4], parameters[i, 5], parameters[i, 6], parameters[i, 0], np.array(range(len(price))), parameters[i, 1], parameters[i, 2], parameters[i, 3])

# Sorting population according to its fitness
print('Sorting Final Population...')
sortf = np.argsort(fit[:, 0])
pop = pop[sortf, :]
parameters = parameters[sortf, :]
fit = fit[sortf]

# Best Result
print('Best Parameters:\ntc:\t', parameters[0, 0], '\nbeta:\t', parameters[0, 1], '\nomega:\t', parameters[0, 2], '\nphi:\t', parameters[0, 3])
print('A:\t', parameters[0, 4], '\nB:\t', parameters[0, 5], '\nC:\t', parameters[0, 6])
print('MAE:\t', fit[0, 0] / len(price))

# Apply Parameters
model = LPPL(parameters[0, 4], parameters[0, 5], parameters[0, 6], parameters[0, 0], np.array(range(len(price))), parameters[0, 1], parameters[0, 2], parameters[0, 3])
# Ploting
# Original
plt.plot(data[:, 1])
# Regression
plt.plot(model)
plt.show()