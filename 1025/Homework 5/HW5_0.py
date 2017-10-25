#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:38:26 2017

@author: sitibanc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv('./TXF20112015.csv', sep=',', header = None)   # dataframe (time, close, open, high, low, volume)
TAIEX = df.values                                               # ndarray
tradeday = list(set(TAIEX[:, 0] // 10000))                      # 交易日（YYYYMMDD）
tradeday.sort()

# Strategy 0.0: 每天第一分鐘開盤價買進，每天最後一分鐘收盤價賣出
profit0 = np.zeros((len(tradeday),1))                            # 當日獲利 
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]           # 取出屬於當日的資料(回傳tuple)
    idx.sort()
    profit0[i] = TAIEX[idx[-1], 1] - TAIEX[idx[0], 2]            # 最後收盤價 - 最初開盤價
profit02 = np.cumsum(profit0)
plt.plot(profit02)                                               # 每日獲利折線圖
print('Strategy 0.0: 每天第一分鐘開盤價買進，每天最後一分鐘收盤價賣出\n每日獲利折線圖')
plt.show()

ans1 = profit02[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit0 > 0) / len(profit0)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit0[profit0 > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit0[profit0 <= 0])                             # 虧損時的平均虧損
plt.hist(profit0, bins = 100)                                    # 每日獲利的分配圖（直方圖）
print('每日獲利分配圖')
plt.show()

# Strategy 0.1: 每天第一分鐘以開盤價空一口台指期，每天最後一分鐘以收盤價買回
profit1 = np.zeros((len(tradeday),1))                            # 當日獲利 
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]           # 取出屬於當日的資料(回傳tuple)
    idx.sort()
    profit1[i] = TAIEX[idx[0], 2] - TAIEX[idx[-1], 1]           # -(最後收盤價 - 最初開盤價)
profit12 = np.cumsum(profit1)
plt.plot(profit12)                                              # 每日獲利折線圖
print('Strategy 0.1: 每天第一分鐘以開盤價空一口台指期，每天最後一分鐘以收盤價買回\n每日獲利折線圖')
plt.show()

ans1 = profit12[-1]                                             # 最後一天的獲利
ans2 = np.sum(profit1 > 0) / len(profit1)                       # 平均獲利（不計入虧損）
ans3 = np.mean(profit1[profit1 > 0])                             # 獲利時的平均獲利
ans4 = np.mean(profit1[profit1 <= 0])                            # 虧損時的平均虧損
plt.hist(profit1, bins = 100)                                   # 每日獲利的分配圖
print('每日獲利分配圖')
plt.show()