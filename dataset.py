import numpy as np
import cv2
import pandas as pd
import os
from keras.utils import to_categorical

# define class encoding
encoding = {'NORMAL': 0, 'PNEUMONIA': 1}

def load_data(type):
    try:
        # load from npz
        X, y = np.load('data/'+type+'.npz')
    except:
        # else load images
        images, labels = [], []
        for root, _, files in os.walk('data/'+type):
            for name in files:
                img = cv2.imread(os.path.join(root+'/'+name))
                if img is not None:
                    resize_img = cv2.resize(img, (224, 224))
                    images.append(resize_img)
                    labels.append(os.path.basename(root))
        images = np.array(images)
        labels = np.array(labels)
        # map to class encoding
        u, ind = np.unique(labels, return_inverse=True)
        y = np.array([encoding[x] for x in u])[ind].reshape(1,-1)
        # map to one hot encoding
        one_hot_y = to_categorical(ind)
        # save to npz and load
        np.savez('data/'+type+'.npz', X=images, y=one_hot_y)
        X, y = np.load('data/'+type+'.npz')
    return X, y

def get_datadict():
    # return all splits in a dict
    for type in ('train', 'test'):
        data['X_'+type], data['y_'+type] = load_data(type)
    return data

def get_datagen():
    # code from Keras Image Preprocessing documentation
    # https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    return datagen
