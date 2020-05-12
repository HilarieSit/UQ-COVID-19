import numpy as np
import cv2
import pandas as pd
import os
from scipy import stats
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def load_data(type):
    try:
        # load from npz
        dataset = np.load('chest_xray/'+type+'.npz')
    except:
        # else load images
        images, labels = [], []
        for root, _, files in os.walk('chest_xray/'+type):
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
        encoding = {'NORMAL': 0, 'PNEUMONIA': 1, 'COVID-19': 2}
        y = np.array([encoding[x] for x in u])[ind]

        # map to one hot encoding
        one_hot_y = to_categorical(y)

        if type != 'test':
            images, y = equate_dataset(images, y)
            np.savez('chest_xray/'+type+'.npz', X=images, y=one_hot_y)
        else:
            np.savez('chest_xray/'+type+'.npz', X=images, y=one_hot_y)

        # save to npz and load
        dataset = np.load('chest_xray/'+type+'.npz')
    return dataset

def get_datadict():
    data = dict()
    # return all splits in a dict
    for type in ('train', 'val', 'test'):
        dataset = load_data(type)
        data['X_'+type], data['y_'+type] = dataset.f.X, dataset.f.y
    return data

def equate_dataset(X, y):
    # random upsampling
    clabels, counts = np.unique(y, return_counts=True)
    inds = np.argsort(-counts)
    max = counts[inds][0]
    for ind in inds:
        clabel = clabels[ind]
        a = counts[ind]
        if a != max:
            size = max - a
            more_ind = np.random.choice(a=a, size=size)
            more_X = X[more_ind,:,:,:]
            more_y = y[more_ind]
            X = np.concatenate((X, more_X), axis=0)
            y = np.concatenate((y, more_y), axis=0)
            print(X.shape)
            print(y.shape)
    return X, y

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

load_data('val')
