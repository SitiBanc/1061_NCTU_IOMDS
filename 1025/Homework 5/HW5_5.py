#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:05:37 2017

@author: sitibanc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Read CSV
# =============================================================================
df = pd.read_csv('TXF20112015.csv', sep=',', header = None)     # dataframe (time, close, open, high, low, volume)
TAIEX = df.values                                               # ndarray
tradeday = list(set(TAIEX[:, 0] // 10000))                      # 交易日（YYYYMMDD）
tradeday.sort()

# =============================================================================
# Strategy 5: 承Strategy 4，加入30點停損點
# =============================================================================
profit0 = np.zeros((len(tradeday),1))
count = 0                                                       # 進場次數
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
    idx.sort()
    openning = TAIEX[idx[0], 2]                                 # 當日開盤價
    long_signal = openning + 30                                 # 買訊
    short_signal = openning - 30                                # 賣訊
    # 符合買訊的時間點
    idx2 = np.nonzero(TAIEX[idx, 3] >= long_signal)[0]          # 買點
    # 設定買訊停損點
    if len(idx2) > 0:
        # 當日交易中在第一個買訊之後（含買訊，故index = 0不能用在停損）的資料
        tmp2 = TAIEX[idx[idx2[0]]:idx[-1], :]
        idx2_stop = np.nonzero(tmp2[:, 4] <= openning)[0]
    # 符合賣訊的時間點
    idx3 = np.nonzero(TAIEX[idx, 4] <= short_signal)[0]         # 賣點
    # 設定賣訊停損點
    if len(idx3) > 0:
        # 當日交易中在第一個賣訊之後（含賣訊，故index = 0不能用在停損）的資料
        tmp3 = TAIEX[idx[idx3[0]]:idx[-1], :]
        idx3_stop = np.nonzero(tmp3[:, 3] >= openning)[0]
        
    if len(idx2) == 0 and len(idx3) == 0:                       # 當日沒有觸及買賣點（不進場）
        p1 = 0
        p2 = 0
    elif len(idx3) == 0:                                        # 當日僅出現買訊（進場做多）
        p1 = TAIEX[idx[idx2[0]], 1]                             # 第一個買點收盤價買進
        if len(idx2_stop) > 1:                                  # 有觸及停損點
            p2 = tmp2[idx2_stop[1], 1]                          # 第一個停損點（index = 1）出現時的收盤價賣出
        else:
            p2 = TAIEX[idx[-1], 1]                              # 當日收盤價賣出
        count += 1
    elif len(idx2) == 0:                                        # 當日僅出現賣訊（進場做空）
        p2 = TAIEX[idx[idx3[0]], 1]                             # 第一個賣點收盤價賣出
        if len(idx3_stop) > 1:                                  # 有觸及停損點
            p1 = tmp3[idx3_stop[1], 1]                          # 停損點出現時的收盤價買回
        else:
            p1 = TAIEX[idx[-1], 1]                              # 當日收盤價買回
        count += 1
    elif idx2[0] > idx3[0]:                                     # 當日賣訊先出現（進場做空）
        p2 = TAIEX[idx[idx3[0]], 1]                             # 第一個賣點收盤價賣出
        if len(idx3_stop) > 1:                                  # 有觸及停損點
            p1 = tmp3[idx3_stop[1], 1]                          # 停損點出現時的收盤價買回
        else:
            p1 = TAIEX[idx[-1], 1]                              # 當日收盤價買回
        count += 1
    else:                                                       # 當日買訊先出現（進場做多）
        p1 = TAIEX[idx[idx2[0]], 1]                             # 第一個買點收盤價買進
        if len(idx2_stop) > 1:                                  # 有觸及停損點
            p2 = tmp2[idx2_stop[1], 1]                          # 停損點出現時的收盤價賣出
        else:
            p2 = TAIEX[idx[-1], 1]                              # 當日收盤價賣出
        count += 1
    profit0[i] = p2 - p1

print('Strategy 5: 承Strategy 4，加入30點停損點\n逐日損益折線圖')
profit02 = np.cumsum(profit0)                                   # 逐日損益獲利
plt.plot(profit02)                                              # 逐日損益折線圖
plt.show()
print('每日損益分佈圖')
plt.hist(profit0, bins = 100)                                   # 每日損益的分佈圖（直方圖）
plt.show()
# 計算數據
ans1 = count                                                    # 進場次數
ans2 = profit02[-1]                                             # 總損益點數
ans3 = np.sum(profit0 > 0) / ans1 * 100                         # 勝率
ans4 = np.mean(profit0[profit0 > 0])                            # 獲利時的平均獲利點數
zero_profit = len(profit0[profit0 <= 0]) - (len(profit0) - ans1)# 進場沒有贏的日數（profit為0 - 沒有進場）
ans5 = np.sum(profit0[profit0 < 0]) / zero_profit               # 虧損時的平均虧損點數
print('進場次數：', ans1, '\n總損益點數：', ans2, '\n勝率：', ans3, '%')
print('賺錢時平均每次獲利點數', ans4, '\n輸錢時平均每次損失點數：', ans5, '\n')
