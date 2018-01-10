# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:02:01 2018

@author: jie-yu
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from PIL import Image
import os

labels = np.load("img_labels.npy")
y_train = np.zeros((len(labels),1))#len(labels)
#def hot_to_num():
for i in range(len(labels)):#len(labels)
    y_train[i] = np.where(labels[i]==1)[0][0]
#image = Image.open("hw3_img.jpg")
os.chdir('D:\\Jie-Yu\\碩一上\\智慧型\\期末project\\img\\img')
filelist = os.listdir()
x = np.zeros((len(filelist),150*150))
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:]=np.array(IMG.getdata())
x_train = x.copy()
x_test = x_train.copy()
y_test = y_train.copy()

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)
# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(150*150,))
# encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(150*150, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)
# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)
"""
Epoch 20/20
60000/60000 [==============================] - 7s - loss: 0.0398
"""
# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()

def lda(X,L):    
    Classes = np.unique(np.array(L))#0,1,2
    k = len(Classes)#k = 3
    n = np.zeros((k,1))#3*1 array
    C = [" "]*k #3*1 list
    M = np.mean(X,axis = 0) #X的mean
    S = [" "]*k #3*1 list
    Sw = 0
    Sb = 0
    for j in range(k):#3
        Xj = X[np.where(L==Classes[j])[0]]
        n[j] = int(Xj.shape[0])
        C[j] = np.mean(Xj,axis = 0)
        S[j] = 0
        for i in range(int(n[j])):
            aaa = np.array([Xj[i,:]-C[j]])
            S[j] = S[j]+np.dot(aaa.T,aaa)
        Sw = Sw+S[j]
        bbb = np.array([C[j]-M])
        Sb = Sb+int(n[j])*np.dot(bbb.T,bbb)
    tmp = np.dot(np.linalg.inv(Sw),Sb)
    LAMBDA,W = np.linalg.eig(tmp)
    SortOrder = np.argsort(-LAMBDA)
#    print(W)
    W = W[:,SortOrder[0:1]]
    Y = np.dot(X,W)
    Y = -Y
    return Y,W
Y,W = lda(encoded_imgs,np.array(y_test))#降成一維的特徵
Y_sort = np.squeeze(Y).argsort()
Y_list = []
for i in range(len(Y_sort)):
    aaa = (x_test[Y_sort[i]]+0.5)*255
    Y_list.append(aaa.reshape(150,150).T.astype('uint8'))
Y_list = np.array(Y_list)
def draw_func(a,b):
    start = min(a,b)
    end = max(a,b)
    if end-start>10:
        jump = (end-start)//10
        draw = Y_list[range(start,end,jump)]
        draw = draw.reshape((len(range(start,end,jump)))*150,150)
    else:
        draw = Y_list[start:end]
        draw = draw.reshape((end-start)*150,150)
    draw = draw.T
    Image.fromarray(draw).show()
    #draw = np.array(Y_list)
draw_func(500,510)
#draw_func(500,502)
#draw_func(502,503)

