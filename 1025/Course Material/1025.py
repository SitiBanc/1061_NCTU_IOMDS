#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:58:01 2017

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
profit = np.zeros((len(tradeday),1))                            # 當日獲利 
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]           # 取出屬於當日的資料(回傳tuple)
    idx.sort()
    profit[i] = TAIEX[idx[-1], 1] - TAIEX[idx[0], 2]            # 最後收盤價 - 最初開盤價
profit2 = np.cumsum(profit)
plt.plot(profit2)                                               # 每日獲利折線圖
plt.show()

ans1 = profit2[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit > 0) / len(profit)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit[profit > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit[profit <= 0])                             # 虧損時的平均虧損
plt.hist(profit, bins = 100)                                    # 每日獲利的分配圖（直方圖）
plt.show()

# Strategy 1.0: 開盤買進一口，30點停損，收盤平倉
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
    profit[i] = p2 - p1
    
profit2 = np.cumsum(profit)
plt.plot(profit2)
plt.show()
ans1 = profit2[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit > 0) / len(profit)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit[profit > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit[profit <= 0])                             # 虧損時的平均虧損
plt.hist(profit, bins = 100)                                    # 每日獲利的分配圖

# Strategy 2.0: 開盤買進一口，30點停損，30點停利，收盤平倉
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]
    # 設定停損點
    idx2 = np.nonzero(TAIEX[idx, 4] <= p1 - 30)[0]              # 最低價跌破停損點
    # 設定停利點
    idx3 = np.nonzero(TAIEX[idx, 3] >= p1 + 30)[0]              # 最高價衝破停利點
    if len(idx2) == 0 and len(idx3) == 0:                       # 當日沒有觸及平損停利點
        p2 = TAIEX[idx[-1], 1]                                  # 當日收盤價賣出
    elif len(idx3) == 0:                                        # 當日沒有停利但停損
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價賣出
    elif len(idx2) == 0:                                        # 當日沒有停損但停利
        p2 = TAIEX[idx[idx3[0]], 1]                             # 停利點收盤價賣出
    elif idx2[0] > idx3[0]:                                     # 當日停利點先出現
        p2 = TAIEX[idx[idx3[0]], 1]                             # 停利點收盤價賣出
    else:                                                       # 當日停損點先出現
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價賣出
    profit[i] = p2 - p1

profit2 = np.cumsum(profit)
plt.plot(profit2)
plt.show()
ans1 = profit2[-1]                                              # 最後一天的獲利
ans2 = np.sum(profit > 0) / len(profit)                         # 平均獲利（不計入虧損）
ans3 = np.mean(profit[profit > 0])                              # 獲利時的平均獲利
ans4 = np.mean(profit[profit <= 0])                             # 虧損時的平均虧損
plt.hist(profit, bins = 100)                                    # 每日獲利的分配圖