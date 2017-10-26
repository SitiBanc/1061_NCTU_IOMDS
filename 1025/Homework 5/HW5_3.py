#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 23:33:46 2017

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
# Strategy 3.0: 開盤買進一口，n點停損，m點停利，收盤平倉，m >= n
# =============================================================================
profit0 = np.zeros((len(tradeday), 1))
tmp_profit0 = np.zeros((len(tradeday), 1))
best0 = [0] * 3                                                 # [n, m, profit]
best = profit0
for n in range(10, 110, 10):
    for m in range(n , n + 100, 10):
        for i in range(len(tradeday)):
            date = tradeday[i]
            idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
            idx.sort()
            p1 = TAIEX[idx[0], 2]
            # 設定停損點
            idx2 = np.nonzero(TAIEX[idx, 4] <= p1 - n)[0]       # 最低價跌破停損點
            # 設定停利點
            idx3 = np.nonzero(TAIEX[idx, 3] >= p1 + m)[0]       # 最高價衝破停利點
            if len(idx2) == 0 and len(idx3) == 0:               # 當日沒有觸及平損停利點
                p2 = TAIEX[idx[-1], 1]                          # 當日收盤價賣出
            elif len(idx3) == 0:                                # 當日沒有停利但停損
                p2 = TAIEX[idx[idx2[0]], 1]                     # 停損點收盤價賣出
            elif len(idx2) == 0:                                # 當日沒有停損但停利
                p2 = TAIEX[idx[idx3[0]], 1]                     # 停利點收盤價賣出
            elif idx2[0] > idx3[0]:                             # 當日停利點先出現
                p2 = TAIEX[idx[idx3[0]], 1]                     # 停利點收盤價賣出
            else:                                               # 當日停損點先出現
                p2 = TAIEX[idx[idx2[0]], 1]                     # 停損點收盤價賣出
            tmp_profit0[i] = p2 - p1
        # 選擇最好的m, n
        if best0[2] < np.sum(tmp_profit0):
            best0[0] = n
            best0[1] = m
            best0[2] = np.sum(tmp_profit0)
            profit0 = tmp_profit0
            best = tmp_profit0
            print('數值更新：', n, m, np.sum(profit0))

print('Strategy 3.0: 當日以開盤價買進一口，', best0[0], '點停損，', best0[1], '點停利，當日收盤價平倉\n逐日損益折線圖')
profit02 = np.cumsum(profit0)                                   # 逐日損益獲利
plt.plot(profit02)                                              # 逐日損益折線圖
plt.show()
print('每日損益分佈圖')
plt.hist(profit0, bins = 100)                                   # 每日損益的分佈圖（直方圖）
plt.show()
# 計算數據
ans1 = len(profit0)                                             # 進場次數
ans2 = profit02[-1]                                             # 總損益點數
ans3 = np.sum(profit0 > 0) / ans1 * 100                         # 勝率
ans4 = np.mean(profit0[profit0 > 0])                            # 獲利時的平均獲利點數
ans5 = np.mean(profit0[profit0 <= 0])                           # 虧損時的平均虧損點數
print('進場次數：', ans1, '\n總損益點數：', ans2, '\n勝率：', ans3, '%')
print('賺錢時平均每次獲利點數', ans4, '\n輸錢時平均每次損失點數：', ans5, '\n')


# =============================================================================
# Strategy 3.1: 開盤賣出一口，n點停損，m點停利，收盤平倉，m >= n
# =============================================================================
#profit1 = np.zeros((len(tradeday),1))
#tmp_profit1 = np.zeros((len(tradeday), 1))
#best1 = [0] * 3
#for n in range(10, 110, 10):
#    for m in range(n , n + 100, 10):
#        for i in range(len(tradeday)):
#            date = tradeday[i]
#            idx = np.nonzero(TAIEX[:, 0] // 10000 == date)[0]
#            idx.sort()
#            p1 = TAIEX[idx[0], 2]
#            # 設定停損點
#            idx2 = np.nonzero(TAIEX[idx, 3] >= p1 + 30)[0]      # 最高價衝破停損點
#            # 設定停利點
#            idx3 = np.nonzero(TAIEX[idx, 4] <= p1 - 30)[0]      # 最低價跌破停利點
#            if len(idx2) == 0 and len(idx3) == 0:               # 當日沒有觸及平損停利點
#                p2 = TAIEX[idx[-1], 1]                          # 當日收盤價買回
#            elif len(idx3) == 0:                                # 當日沒有停利但停損
#                p2 = TAIEX[idx[idx2[0]], 1]                     # 停損點收盤價買回
#            elif len(idx2) == 0:                                # 當日沒有停損但停利
#                p2 = TAIEX[idx[idx3[0]], 1]                     # 停利點收盤價買回
#            elif idx2[0] > idx3[0]:                             # 當日停利點先出現
#                p2 = TAIEX[idx[idx3[0]], 1]                     # 停利點收盤價買回
#            else:                                               # 當日停損點先出現
#                p2 = TAIEX[idx[idx2[0]], 1]                     # 停損點收盤價買回
#            tmp_profit1[i] = p1 - p2
#        # 選擇最好的m, n
#        if best1[2] < np.sum(tmp_profit1):
#            best1[0] = n
#            best1[1] = m
#            best1[2] = np.sum(tmp_profit1)
#            profit1 = tmp_profit1
#
#print('Strategy 3.1: 當日以開盤價賣出一口，', best1[0], '點停損，', best1[1], '點停利，當日收盤價平倉\n逐利損益折線圖')
#profit12 = np.cumsum(profit1)                                   # 逐日累積損益
#plt.plot(profit12)                                              # 逐日損益折線圖
#plt.show()
#print('每日損益分佈圖')
#plt.hist(profit1, bins = 100)                                   # 每日損益的分佈圖
#plt.show()
## 計算數據
#ans1 = len(profit1)                                             # 進場次數
#ans2 = profit12[-1]                                             # 總損益點數
#ans3 = np.sum(profit1 > 0) / ans1 * 100                         # 勝率
#ans4 = np.mean(profit1[profit1 > 0])                            # 獲利時的平均獲利點數
#ans5 = np.mean(profit1[profit1 <= 0])                           # 虧損時的平均虧損點數
#print('進場次數：', ans1, '\n總損益點數：', ans2, '\n勝率：', ans3, '%')
#print('賺錢時平均每次獲利點數', ans4, '\n輸錢時平均每次損失點數：', ans5)