import csv
import numpy as np
import pandas as pd
from PIL import Image
import os

def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for (root, dirs, files) in os.walk(myDir, topdown=True):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root,name)
                fileList.append(fullName)
    return fileList

myFileList = createFileList('dataset/SmallestHands')
print(myFileList)

#make header for csv
arr = np.arange(30000)
arr = np.append(arr, "target")
with open("dataset/img_pixels3.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(arr)

hInfo = pd.read_csv('dataset/HandInfo.csv')
id = hInfo.id
aspect = hInfo.aspectOfHand


i = 0
for file in myFileList:
    #print(file)
    if aspect[i] == "dorsal right" or aspect[i] == "dorsal left":
        img_file = Image.open(file)

        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode

        img_grey=img_file.convert('L')

        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
        value = value.flatten()
        value = np.append(value, id[i])

        #print(value)

        with open("dataset/img_pixels3.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
    i = i + 1
