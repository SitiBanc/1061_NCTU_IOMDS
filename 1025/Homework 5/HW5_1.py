#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:51:00 2017

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

# Strategy 1.0: 每日第一分鐘開盤買進一口，30點停損，每天最後一分鐘以收盤價平倉
profit0 = np.zeros((len(tradeday),1))                            # 當日獲利
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]                                       # 最初開盤價買入
    # 設定停損點
    idx2 = np.nonzero(TAIEX[idx, 4] <= p1 - 30)[0]              # 最低價跌破停損點
    if len(idx2) == 0:                                          # 沒有跌破
        p2 = TAIEX[idx[-1], 1]                                  # 當日收盤價賣出
    else:                                                       # 最低價跌破停損點
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價賣出
    profit0[i] = p2 - p1
    
profit02 = np.cumsum(profit0)
ans1 = profit02[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit0 > 0) / len(profit0)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit0[profit0 > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit0[profit0 <= 0])                             # 虧損時的平均虧損
print('Strategy 1.0: 每日第一分鐘開盤買進一口，30點停損，每天最後一分鐘以收盤價平倉\n每日獲利折線圖')
plt.plot(profit02)
plt.show()
print('每日獲利分配圖')
plt.hist(profit0, bins = 100)                                    # 每日獲利的分配圖
plt.show()

# Strategy 1.1: 每日第一分鐘開盤空一口，30點停損，每天最後一分鐘以收盤價平倉
profit1 = np.zeros((len(tradeday),1))                            # 當日獲利
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]                                       # 最初開盤價賣出
    # 設定停損點
    idx2 = np.nonzero(TAIEX[idx, 4] >= p1 + 30)[0]              # 最高價衝破停損點
    if len(idx2) == 0:                                          # 沒有跌破
        p2 = TAIEX[idx[-1], 1]                                  # 當日收盤價賣出
    else:                                                       # 最低價跌破停損點
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價賣出
    profit1[i] = p1 - p2
    
profit12 = np.cumsum(profit1)
ans1 = profit12[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit1 > 0) / len(profit1)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit1[profit1 > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit1[profit1 <= 0])                             # 虧損時的平均虧損
print('Strategy 1.0: 開盤買進一口，30點停損，收盤平倉\n每日獲利折線圖')
plt.plot(profit12)
plt.show()
print('每日獲利分配圖')
plt.hist(profit1, bins = 100)                                    # 每日獲利的分配圖
plt.show()