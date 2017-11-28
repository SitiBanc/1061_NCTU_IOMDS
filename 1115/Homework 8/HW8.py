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
    return LPPL(A, B, C, tc, t, beta, omega, phi) - ans


# Log-Periodic Power Laws for bubble modeling
def LPPL(A, B, C, tc, t, beta, omega, phi):
    return A + (B * np.power(tc - t, beta)) * (1 + (C * np.cos((omega * np.log(tc - t)) + phi)))

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
price = data[:500, 1]
# Genetic Algorithm
# Genes of Initial Population
pop = np.random.randint(0, 2, (10000, 25))
fit = np.zeros((10000, 1))
# Parameters
parameters = np.zeros((10000, 4))


for generation in range(100):
    print('---------- Generation', generation + 1, '----------')
    # Calculate Fitness using Energy Function
    for i in range(10000):
        print('Genes Decoding...')
        gene = pop[i, :]
        # binary to decimal
        # tc: 7 bits, range from 500 ~ 627
        parameters[i, 0] = np.sum(2 ** np.array(range(7)) * gene[:7]) + 500
        # beta: 8 bits, range from 1/257 ~ 256/257
        parameters[i, 1] = (np.sum(2 ** np.array(range(8)) * gene[7:15]) + 1) / 257
        # omega: 4 bits, range from 4 ~ 19
        parameters[i, 2] = np.sum(2 ** np.array(range(4)) * gene[15:19]) + 4
        # phi: 6 bits, range from 0.01 ~ 63.02/2pi
        parameters[i, 3] = (np.sum(2 ** np.array(range(6)) * gene[19:]) + 0.01) / (63.02 / (np.pi * 2))
        print('Linear Regression...')
        # Linear Regression
        items = np.zeros((500, 3))
        # A's coefficient
        items[:, 0] += 1
    #    for j in range(500):
    #        # A's coefficient
    #        items[j, 0] = 1
    #        # B's coefficient
    #        items[j, 1] = np.power(parameters[:, 0] - j, parameters[i, 1])
    #        # BC's coefficient
    #        items[j, 2] = items[j, 1] * np.cos((parameters[i, 2] * np.log(parameters[i, 0] - j)) + parameters[i, 3])
        # A's coefficient
        items[:, 0] += 1
        # B's coefficient
        items[:, 1] += np.power(parameters[:, 0] - j, parameters[:, 1])
        # BC's coefficient
        items[:, 2] += items[j, 1] * np.cos((parameters[:, 2] * np.log(parameters[:, 0] - j)) + parameters[:, 3])
        # Least Square
        x = np.linalg.lstsq(items, price)[0]
        A = x[0, 0]
        B = x[1, 0]
        C = x[2, 0] / B
        print('Calculate Fitness...')
        fit[i] = E2()
        
    # Elimination
    # Using Tourment Select to decide survival
    # Top 100 survived, others will be replaced during crossover
    # Sorting population according to its fitness
    print('Eliminating...')
    sortf = np.argsort(fit[:, 0])
    pop = pop[sortf, :]
    
    # Reproduction
    # Crossover using mask (0: mother; 1: father)
    # Top 100 survived, crossover till population is filled
    print('Reproducing...')
    for i in range(100, 10000):
        # fid: father id; mid: mother id
        fid = np.random.randint(0, 100)
        mid = np.random.randint(0, 100)
        while mid == fid:
            mid = np.random.randint(0, 100)
        mask = np.random.randint(0, 2, (1, 25))
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
        m = np.random.randint(0, 10000)
        # the mutated genes
        n = np.random.randint(0, 25)
        # Switch genes
        if pop[m, n] == 1:
            pop[m, n] = 0
        else:
            pop[m, n] = 1

# Calculate Fitness
print('---------- Final Result ----------')
print('Calculate Final Fitness...')
for i in range(10000):
    gene = pop[i, :]
    # binary to decimal
    tc = np.sum(2 ** np.array(range(7)) * gene[:7]) + 500
    beta = (np.sum(2 ** np.array(range(8)) * gene[7:15]) + 1) / 257
    omega = np.sum(2 ** np.array(range(4)) * gene[15:19]) + 4
    phi = np.sum(2 ** np.array(range(6)) * gene[19:]) + 0.01 / (63.02 / np.pi * 2)
    fit[i] = E(data[:, 1], ts, tc, beta, omega, phi)

# Sorting population according to its fitness
print('Sorting Final Population...')
sortf = np.argsort(fit[:, 0])
pop = pop[sortf, :]

# Best Result
best_gene = pop[0, :]
# binary to decimal
best_tc = np.sum(2 ** np.array(range(7)) * best_gene[:7]) + 500
best_beta = (np.sum(2 ** np.array(range(8)) * best_gene[7:15]) + 1) / 257
best_omega = np.sum(2 ** np.array(range(4)) * best_gene[15:19]) + 4
best_phi = (np.sum(2 ** np.array(range(6)) * best_gene[19:]) + 0.01) / (63.02 / (np.pi * 2))
best_fitness = E(targets, data[:, 1], tc, beta, omega, phi)
print('Best Parameters:\ntc:\t', best_tc, '\nbeta:\t', best_beta, '\nomega:\t', best_omega, '\nphi:\t', best_phi)