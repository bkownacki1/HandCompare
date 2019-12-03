from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC


#shows original images in greyscale reconstructed from csv data
def showOriginalImages(pixels):
    fig, axes = plt.subplots(20, 5, figsize=(20, 15),
                             subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(150,200), cmap='gray')
    plt.show()

#shows eigenspace from hand data
def showEigenHands(pca):
    fig, axes = plt.subplots(20, 5, figsize=(7.5, 20),
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(wspace=0)
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(150, 200), cmap='gray')
        #ax.set_title("PC " + str(i+1))
    plt.show()

# def plotSVC(title, X):
#      #create a mesh to plot in
#      y = X.target
#      x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#      y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#      h = (x_max / x_min)/100
#      xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#      np.arange(y_min, y_max, h))
#      plt.subplot(1, 1, 1)
#      Z = SVC.predict(np.c_[xx.ravel(), yy.ravel()])
#      Z = Z.reshape(xx.shape)
#      plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#      plt.scatter(X[:, 0], X[:, 1], c = y, cmap=plt.cm.Paired)
#      plt.xlabel('Sepal length')
#      plt.ylabel('Sepal width')
#      plt.xlim(xx.min(), xx.max())
#      plt.title(title)
#      plt.show()

print("running")

maxChunksAllowed = 650
optimalNValue = 250
j = 0
running = True

for chunk in pd.read_csv("dataset/img_pixels3.csv", chunksize=maxChunksAllowed, dtype=object, low_memory=False):
    labels = chunk["target"]
    pixels = chunk.drop(["target"], axis=1)
    print("labels and pixels assigned")

    #shows original images
    #showOriginalImages(pixels)

    print("assigning train test split values..")
    x_train, x_test, y_train, y_test = train_test_split(pixels, labels)
    print("train test split values assigned")

    print("running pca with " + str(optimalNValue) + " n_components fitting to x_train...")
    pca = PCA(n_components=optimalNValue).fit(x_train)
    print("pca finished")

    #shows graph to determine optimal n_component value
    # print("plotting results...")
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.show()
    # print("plotted.")

    #shows eigengraph
    # print("visualizing eigenhands...")
    # showEigenHands(pca)
    # print("visualized.")

    #train model using PCA
    print("training pca...")
    x_train_pca = pca.transform(x_train)
    print("finished training pca")

    #classify with SVC using polynomial of degree 4 with gamma of 1
    print("making classifier...")
    clf = SVC(kernel='poly', C=100, gamma=1, degree=4)
    print("fitting classifier...")
    clf = clf.fit(x_train_pca, y_train)

    #test model
    x_test_pca = pca.transform(x_test)
    print("making predictions...")
    y_pred = clf.predict(x_test_pca)

    #print performance
    print(classification_report(y_test, y_pred))

    #plotSVC('graph with poly4, c=100, & gamma=1', clf)

    break


print("program complete")
