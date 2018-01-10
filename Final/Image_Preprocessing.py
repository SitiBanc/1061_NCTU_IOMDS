#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:09:21 2018

@author: sitibanc
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image


train_data_dir = '/home/sitibanc/wikiart'
output_dir = '/home/sitibanc/1061_NCTU_IOMDS/Final/rgb_img'
img_height, img_width = 150, 150
batch_size = 200

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='rgb',
        save_to_dir=output_dir,
        save_prefix='img',  # partial control over filenames
        save_format='jpeg')

# Print Classes
#print(train_generator.class_indices)

nb_train_samples = 5400
nb_classes = len(train_generator.class_indices)
labels = np.zeros((nb_train_samples, nb_classes))   # One-hot encoded
img_id = 0

for i in range(nb_train_samples // batch_size):
    print('Batch:\t%d' % (i))
    x, y = train_generator.next()
    labels[i*batch_size:(i + 1)*batch_size, :] = y
    # Use PIl to save output files (fully control filenames)
#    for j in range(len(x)):
#        file_name = '%d_images/b%d_img%d.jpeg' % (img_id, i, j)
#        Image.fromarray(x[j,:,:,:].astype('uint8')).save(file_name)
#        img_id += 1