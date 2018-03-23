# pylint: disable = C0103, C0111, C0301
import math
import numpy as np
from PIL import Image, ImageDraw

npzfile = np.load('CBCL.npz')   # npzfile.files為npz內所有欄位
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
trpn = trainface.shape[0]   # train positive number
trnn = trainnonface.shape[0]   # train negative number

def faceNum():
    fn = 0   # face number
    ftable = []
    for y in range(19):   # 在19*19的框框裡區塊大小至少為2*2
        for x in range(19):
            for h in range(2, 20):   # 區塊寬(row)
                for w in range(2, 20):   # 區塊長(column)
                    if y + h * 1 - 1 <= 18 and x + w * 2 - 1 <= 18:   # 比較左右兩個相鄰區塊的次數
                        fn += 1
                        ftable.append([0, y, x, h, w])
                    if y + h * 2 - 1 <= 18 and x + w * 1 - 1 <= 18:   # 比較上下兩個相鄰區塊的次數
                        fn += 1
                        ftable.append([1, y, x, h, w])
                    if y + h * 1 - 1 <= 18 and x + w * 3 - 1 <= 18:   # 比較左右三個相鄰區塊的次數，可凸顯眼睛特徵
                        fn += 1
                        ftable.append([2, y, x, h, w])
                    if y + h * 2 - 1 <= 18 and x + w * 2 - 1 <= 18:   # 比較田字四個相鄰區塊的次數，可凸顯臉頰弧度特徵
                        fn += 1
                        ftable.append([3, y, x, h, w])
    print('training face feature number:', fn)
    return fn, ftable

def integralImage(image):   # integral image
    i = image.shape[1]   # width
    j = image.shape[0]   # height
    matrix = np.zeros((j + 1, i + 1), int)
    for m in range(1, j + 1):
        for n in range(1, i + 1):
            matrix[m][n] = image[m - 1][n - 1] + matrix[m][n - 1]

    for m in range(1, j + 1):
        for n in range(1, i + 1):
            matrix[m][n] += matrix[m - 1][n]
    return matrix

def sampleFeature(state, sample, ftable, c, ratio): # sample為n*361的矩陣，計算每個特徵區塊顏色的差值(取特徵用)
    ftype, y, x, h, w = ftable[c][0], ftable[c][1], ftable[c][2], ftable[c][3]*ratio, ftable[c][4]*ratio
    if state == 'train':
        T = np.arange(sample.shape[1]).reshape((19*ratio, 19*ratio))   # 紀錄原圖index，用來方便矩陣取值
        if ftype == 0:
            output = np.sum(sample[:, T[y: y+h, x: x+w].flatten()], axis=1) - np.sum(sample[:, T[y: y+h, x+w: x+w*2].flatten()], axis=1)   # 白-黑
        elif ftype == 1:
            output = -np.sum(sample[:, T[y: y+h, x: x+w].flatten()], axis=1) + np.sum(sample[:, T[y+h: y+h*2, x: x+w].flatten()], axis=1)   # 白-黑
        elif ftype == 2:
            output = np.sum(sample[:, T[y: y+h, x: x+w].flatten()], axis=1) - np.sum(sample[:, T[y: y+h, x+w: x+w*2].flatten()], axis=1) + np.sum(sample[:, T[y: y+h, x+w*2: x+w*3].flatten()], axis=1)   # 白-黑+白
        elif ftype == 3:
            output = np.sum(sample[:, T[y: y+h, x: x+w].flatten()], axis=1) - np.sum(sample[:, T[y: y+h, x+w: x+w*2].flatten()], axis=1) - np.sum(sample[:, T[y+h: y+h*2, x: x+w].flatten()], axis=1) + np.sum(sample[:, T[y+h: y+h*2, x+w: x+w*2].flatten()], axis=1)   # 白-黑-黑+白
    elif state == 'test':
        T = np.arange(sample.shape[1]).reshape((19*ratio+1, 19*ratio+1))
        if ftype == 0:
            output = sample[:, T[y, x]] + sample[:, T[y+h, x+w]] - sample[:, T[y, x+w]] - sample[:, T[y+h, x]] - (sample[:, T[y, x+w]] + sample[:, T[y+h, x+w*2]] - sample[:, T[y, x+w*2]] - sample[:, T[y+h, x+w]])
        elif ftype == 1:
            output = - (sample[:, T[y, x]] + sample[:, T[y+h, x+w]] - sample[:, T[y, x+w]] - sample[:, T[y+h, x]]) + (sample[:, T[y+h, x]] + sample[:, T[y+h*2, x+w]] - sample[:, T[y+h, x+w]] - sample[:, T[y+h*2, x]])
        elif ftype == 2:
            output = sample[:, T[y, x]] + sample[:, T[y+h, x+w]] - sample[:, T[y, x+w]] - sample[:, T[y+h, x]] - (sample[:, T[y, x+w]] + sample[:, T[y+h, x+w*2]] - sample[:, T[y, x+w*2]] - sample[:, T[y+h, x+w]]) + (sample[:, T[y, x+w*2]] + sample[:, T[y+h, x+w*3]] - sample[:, T[y, x+w*3]] - sample[:, T[y+h, x+w*2]])
        elif ftype == 3:
            output = sample[:, T[y, x]] + sample[:, T[y+h, x+w]] - sample[:, T[y, x+w]] - sample[:, T[y+h, x]] - (sample[:, T[y, x+w]] + sample[:, T[y+h, x+w*2]] - sample[:, T[y, x+w*2]] - sample[:, T[y+h, x+w]]) - (sample[:, T[y+h, x]] + sample[:, T[y+h*2, x+w]] - sample[:, T[y+h, x+w]] - sample[:, T[y+h*2, x]]) + (sample[:, T[y+h, x+w]] + sample[:, T[y+h*2, x+w*2]] - sample[:, T[y+h, x+w*2]] - sample[:, T[y+h*2, x+w]])
    return output   # shape = (sample.shape[0], 1) h*w累加出一個值

def weakClassifier(pw, nw, pf, nf):   # weak classifier
    maxf = max(pf.max(), nf.max())   # max feature
    minf = min(pf.min(), nf.min())
    theta = (maxf - minf) / 10 + minf   # 第1刀
    error = np.sum(pw[pf < theta]) + np.sum(nw[nf >= theta])   # 計算所有pf比theta小的error
    polarity = 1   # 右正左負
    if error > 0.5:   # 反過來猜
        error = 1 - error
        polarity = 0

    min_theta, min_error, min_polarity = theta, error, polarity
    for i in range(2, 10):   # 第2~10刀
        theta = (maxf - minf) * i / 10 + minf   # 等切成10刀中的第一刀(用刀切右正左負)
        error = np.sum(pw[pf < theta]) + np.sum(nw[nf >= theta])   # 計算所有pf比theta小的error
        polarity = 1
        if error > 0.5:   # 反過來猜
            error = 1 - error
            polarity = 0
        if error < min_error:
            min_theta, min_error, min_polarity = theta, error, polarity
    return min_error, min_theta, min_polarity

def strongClassifier(trpf, trnf, fn, iteration):
    pw = np.ones((trpn, 1)) / trpn / 2   # positive weight總和0.5
    nw = np.ones((trnn, 1)) / trnn / 2   # negative weight總和0.5
    SC = []   # strong classifier
    for t in range(iteration):   # 取20個最具鑑別度的特徵
        weightsum = np.sum(pw) + np.sum(nw)
        pw = pw / weightsum
        nw = nw / weightsum
        best_error, best_theta, best_polarity = weakClassifier(pw, nw, trpf[:, 0], trnf[:, 0])   # 輸入第1個特徵
        for i in range(1, fn):   # 跑所有特徵
            error, theta, polarity = weakClassifier(pw, nw, trpf[:, i], trnf[:, i])
            if error < best_error:
                best_feature, best_error, best_theta, best_polarity = i, error, theta, polarity
        beta = best_error / (1 - best_error)   # 小於1，分對的權重降低
        alpha = math.log10(1 / beta)
        SC.append([best_feature, best_theta, best_polarity, alpha])   # 選取第SC[0, 0]個特徵，大於SC[0, 1]，分法，得分
        if best_polarity == 1:   # 右正左負分對的，分錯的權重不變
            pw[trpf[:, best_feature] >= best_theta] *= beta
            nw[trnf[:, best_feature] < best_theta] *= beta   # trnf看成trpf負資料
        else:   # 左正右負分對的，分錯的權重不變
            pw[trpf[:, best_feature] < best_theta] *= beta
            nw[trnf[:, best_feature] >= best_theta] *= beta
    print('Strong Classifier:', SC)
    return SC

def faceScore(bestFilter, SC, trf, trn, threshold, ratio, pixel):
    trs = np.zeros((trn, 1))   # train score
    alpha_sum = 0
    for i in range(bestFilter):
        theta, polarity, alpha = SC[i][1]*ratio, SC[i][2], SC[i][3]
        alpha_sum += alpha
        if polarity == 1:   # 右正左負，只要是正(右邊)的都紀錄累加分數
            trs[trf[:, i] >= theta] += alpha
        else:   # 左正右負，只要是正(左邊)的都紀錄累加分數
            trs[trf[:, i] < theta] += alpha
    trs = trs / alpha_sum
    print('%dx%d filter size, threshold %.3f, jump %d pixel(s), %d faces.' % (19*ratio, 19*ratio, threshold, pixel, np.sum(trs > threshold)))   # 可再用threshold分人臉，大於為人臉
    return trs, np.where(trs > threshold)[0]   # 回傳大於threshold的index

# train strong classifier
#facenum, facetable = faceNum()
#trpf = np.zeros((trpn, facenum))   # train positive faces, iterate每張圖每個特徵的顏色差值
#trnf = np.zeros((trnn, facenum))   # train negative faces
#for c in range(facenum):
#    trpf[:, c] = sampleFeature('train', trainface, facetable, c, 1)
#    trnf[:, c] = sampleFeature('train', trainnonface, facetable, c, 1)
bestFilter = 20
#SC = strongClassifier(trpf, trnf, facenum, bestFilter)

# save = [[SC, facetable], ['sc', 'ftable']]
# for i in range(len(save[0])):   # 寫檔
#     f = open(save[1][i] + '.txt', 'w')
#     for m in save[0][i]:
#         f.write(' '.join([str(n) for n in m]) + '\n')
#     f.close()
File = []
File.append([i.split(' ') for i in open('sc.txt', 'r').readlines()])   # 讀檔
SC = File[0]
SC = list(map(lambda i: [int(SC[i][0]), float(SC[i][1]), int(SC[i][2]), float(SC[i][3])], range(20)))
File = []
File.append([i.split(' ') for i in open('ftable.txt', 'r').readlines()])   # 讀檔
facetable = np.asarray(File[0], dtype=int).tolist()

img = Image.open('2.jpg')
print('image size:', img.size)
grey_img = img.convert('L')   # 圖片轉灰階
img_arr = np.array(grey_img).astype(int)   # tuple轉array後再把uint8轉int32
integImg = integralImage(img_arr)
draw = ImageDraw.Draw(img)

thresholds = [0.8, 0.8, 0.82, 0.83, 0.825, 0.795, 0.8, 0.79, 0.773, 0.778]   # 不同大小filter的門檻值
color = ['#FF0000', '#FFFF00', '#77FF00', '#00FFFF', '#FFFFFF', '#FF0000', '#FFFF00', '#77FF00', '#00FFFF', '#FFFFFF']   # 紅黃綠藍白 # 色碼表 https://www.toodoo.com/db/color.html 全系列 https://www.toolskk.com/color.php
jumpPixel = np.array([1, 1, 1, 1, 1, 6, 8, 8, 10, 10], int)   # 不同大小filter的jump pixel個數
for r in range(1, 2):   # filter放大比率
    filterSize = 19 * r
    heightNum, widthNum = (img_arr.shape[0]-filterSize+1) // jumpPixel[r-1], (img_arr.shape[1]-filterSize+1) // jumpPixel[r-1]
    imageBlock = np.zeros((heightNum * widthNum, (filterSize)**2))   # for暴力解，人臉個數*人臉大小拉平
    #imageBlock = np.zeros((heightNum * widthNum, (filterSize+1)**2))   # for integral image，人臉個數*人臉大小拉平
    position = []   # 對應: 大圖(j, i) 臉(y, x) 特徵filter(h, w)
    for j in range(heightNum):   # 短邊
        for i in range(widthNum):   # 長邊
            tmpi, tmpj = i*jumpPixel[r-1], j*jumpPixel[r-1]
            imageBlock[j*widthNum+i, :] = img_arr[tmpj: tmpj+filterSize, tmpi: tmpi+filterSize].flatten()   # for暴力解
            #imageBlock[j*widthNum+i, :] = integImg[tmpj: tmpj+filterSize+1, tmpi: tmpi+filterSize+1].flatten()   # for integral image
            position.append([tmpj, tmpi])

    trf = np.zeros((imageBlock.shape[0], bestFilter))   # trace image faces
    for c in range(bestFilter):
        trf[:, c] = sampleFeature('train', imageBlock, facetable, SC[c][0], r)
    trs, faceIdx = faceScore(bestFilter, SC, trf, imageBlock.shape[0], thresholds[r-1], r, jumpPixel[r-1]-1)

    for k in range(faceIdx.shape[0]):   # 標記人臉位置
        facej, facei = position[faceIdx[k]][0], position[faceIdx[k]][1]
        draw.rectangle([facei, facej, facei+filterSize, facej+filterSize], outline=color[r-1])
img.show()
#img.save(str(img.size[0]) + 'x' + str(img.size[1]) + '.jpg')
