import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

def load_data(type, enc):
    try:
        X, y = np.load('data/'+type+'.npz')
    except:
        images, labels = [], []
        for root, _, files in os.walk('data/'+type):
            for name in files:
                img = cv2.imread(os.path.join(root+'/'+name))
                if img is not None:
                    resize_img = cv2.resize(img, (224, 224))
                    images.append(resize_img)
                    labels.append(os.path.basename(root))
                    break
        images = np.array(images)
        labels = np.array(labels).reshape(1,-1)
        if type == 'train':
            y = enc.fit_transform(labels).toarray()
        else:
            y = enc.transform(labels).toarray()
        np.savez('data/'+type+'.npz', X=images, y=y)
        X, y = np.load('data/'+type+'.npz')
    return X, y

def return_Xy():
    X, y = dict(), dict()
    enc = OneHotEncoder()
    for type in ('train', 'val', 'test'):
        X[type], y[type] = load_data(type, enc)
    return X, y

X, y = return_Xy()
print(X['train'])
