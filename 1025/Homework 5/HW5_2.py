#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:46:31 2017

@author: sitibanc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Read CSV
# =============================================================================
df = pd.read_csv('./TXF20112015.csv', sep=',', header = None)   # dataframe (time, close, open, high, low, volume)
TAIEX = df.values                                               # ndarray
tradeday = list(set(TAIEX[:, 0] // 10000))                      # 交易日（YYYYMMDD）
tradeday.sort()

# =============================================================================
# Strategy 2.0: 開盤買進一口，30點停損，30點停利，收盤平倉
# =============================================================================
profit0 = np.zeros((len(tradeday),1))
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
    profit0[i] = p2 - p1

print('Strategy 2.0: 當日以開盤價買進一口，30點停損，30點停利，當日收盤價平倉\n逐日損益折線圖')
profit02 = np.cumsum(profit0)                                   # 逐日損益獲利
plt.plot(profit02)                                              # 逐日損益折線圖
plt.show()
print('每日損益分佈圖')
plt.hist(profit0, bins = 100)                                   # 每日損益的分佈圖（直方圖）
plt.show()
# 計算數據
ans1 = len(profit0)                                             # 進場次數
ans2 = profit02[-1]                                             # 總損益點數
ans3 = np.sum(profit0 > 0) / len(profit0) * 100                 # 勝率
ans4 = np.mean(profit0[profit0 > 0])                            # 獲利時的平均獲利點數
ans5 = np.mean(profit0[profit0 <= 0])                           # 虧損時的平均虧損點數
print('進場次數：', ans1, '\n總損益點數：', ans2, '\n勝率：', ans3, '%')
print('賺錢時平均每次獲利點數', ans4, '\n輸錢時平均每次損失點數：', ans5, '\n')


# =============================================================================
# Strategy 2.1: 開盤賣出一口，30點停損，30點停利，收盤平倉
# =============================================================================
profit1 = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    p1 = TAIEX[idx[0], 2]
    # 設定停損點
    idx2 = np.nonzero(TAIEX[idx, 3] >= p1 + 30)[0]              # 最高價衝破停損點
    # 設定停利點
    idx3 = np.nonzero(TAIEX[idx, 4] <= p1 - 30)[0]              # 最低價跌破停利點
    if len(idx2) == 0 and len(idx3) == 0:                       # 當日沒有觸及平損停利點
        p2 = TAIEX[idx[-1], 1]                                  # 當日收盤價買回
    elif len(idx3) == 0:                                        # 當日沒有停利但停損
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價買回
    elif len(idx2) == 0:                                        # 當日沒有停損但停利
        p2 = TAIEX[idx[idx3[0]], 1]                             # 停利點收盤價買回
    elif idx2[0] > idx3[0]:                                     # 當日停利點先出現
        p2 = TAIEX[idx[idx3[0]], 1]                             # 停利點收盤價買回
    else:                                                       # 當日停損點先出現
        p2 = TAIEX[idx[idx2[0]], 1]                             # 停損點收盤價買回
    profit1[i] = p1 - p2

print('Strategy 2.1: 當日以開盤價賣出一口，30點停損，30點停利，當日收盤價平倉\n每日獲利折線圖')
profit12 = np.cumsum(profit1)                                   # 逐日累積損益
plt.plot(profit12)                                              # 逐日損益折線圖
plt.show()
print('每日損益分佈圖')
plt.hist(profit1, bins = 100)                                   # 每日損益的分佈圖
plt.show()
# 計算數據
ans1 = len(profit1)                                             # 進場次數
ans2 = profit12[-1]                                             # 總損益點數
ans3 = np.sum(profit1 > 0) / len(profit1) * 100                 # 勝率
ans4 = np.mean(profit1[profit1 > 0])                            # 獲利時的平均獲利點數
ans5 = np.mean(profit1[profit1 <= 0])                           # 虧損時的平均虧損點數
print('進場次數：', ans1, '\n總損益點數：', ans2, '\n勝率：', ans3, '%')
print('賺錢時平均每次獲利點數', ans4, '\n輸錢時平均每次損失點數：', ans5)