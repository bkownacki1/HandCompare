from __future__ import print_function

import keras
import os
import imageio
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

naming_dict={}
f = open("HandInfoStripped.txt", "r")
fileContents = f.read()
fileContents = fileContents.split('\t')
for i in range(len(fileContents)-1):
    fileContents[i] = fileContents[i].split(',')
    naming_dict[fileContents[i][0]] = fileContents[i][1]



batch_size = 1000
num_classes = 2
epochs = 12

img_rows, img_cols = 1200, 1600

(x_train, y_train), (x_test,y_test) = 0

print("")
