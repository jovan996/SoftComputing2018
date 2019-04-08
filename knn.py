import os
import cv2
import numpy as np
from sklearn.datasets import fetch_mldata

knn = cv2.ml.KNearest_create()

def train_knn():

    if os.path.isfile('knn_data.npz'):
        with np.load('knn_data.npz') as filedata:
            
            data = filedata['train']
            labels = filedata['train_labels']
            knn.train(data.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32))
        print("Loaded KNN data from disk")
    else:
        print("KNN train begin")

        mnist = fetch_mldata("MNIST original")
        data = (mnist.data / 255.0) > 0.5
        labels = mnist.target.astype('int')

        for var in range(len(data)):
            img = data[var]
            img = img.reshape(28, 28)
            img = transform_img(img)
            data[var] = img.flatten()

        print("KNN train finish")

        np.savez('knn_data.npz', train=data, train_labels=labels)

        print("Saved KNN data to disk")
