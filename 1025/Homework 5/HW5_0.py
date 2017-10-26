#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:38:26 2017

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
# Strategy 0.0: 每天第一分鐘開盤價買進，每天最後一分鐘收盤價賣出
# =============================================================================
profit0 = np.zeros((len(tradeday),1))                           # 當日損益 
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]           # 取出屬於當日的資料(回傳tuple)
    idx.sort()
    profit0[i] = TAIEX[idx[-1], 1] - TAIEX[idx[0], 2]           # 最後收盤價 - 最初開盤價
# 畫圖
print('Strategy 0.0: 每天第一分鐘開盤價買進，每天最後一分鐘收盤價賣出\n逐日累積損益折線圖')
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
# Strategy 0.1: 每天第一分鐘以開盤價空一口台指期，每天最後一分鐘以收盤價買回
# =============================================================================
profit1 = np.zeros((len(tradeday),1))                           # 當日損益 
for i in range(len(tradeday)):
    date = tradeday[i]
    idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]           # 取出屬於當日的資料
    idx.sort()
    profit1[i] = TAIEX[idx[0], 2] - TAIEX[idx[-1], 1]           # -(最後收盤價 - 最初開盤價)
# 畫圖
print('Strategy 0.1: 每天第一分鐘開盤價賣出，每天最後一分鐘收盤價買回\n逐日累積損益折線圖')
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