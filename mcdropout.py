import keras
from keras import models
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras import callbacks
import numpy as np
import dataset

# check if using gpu
import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
import seaborn as sns

NUM_CLASSES = 2         # number of classes
NUM_MC = 10           # number of monte carlo samples

class MCDropout:
    def __init__(self, filepath, datagen=None, data=None):
        try:
            self.model = models.load_model(filepath+'model.h5')
        except:
            self.data = data
            self.filepath = filepath
            self.model = self.train(datagen)

    def train(self, datagen):
        # architecture
        input = Input(shape=(224, 224 ,3))
        # default is channels last
        x = Conv2D(16, kernel_size=(3, 3), activation='relu')(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x, training=True)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x, training=True)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)

        # optimizer/loss
        optimizer = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999,
            amsgrad=False)
        model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=optimizer, metrics=['accuracy'])
        print(self.filepath+'model.h5')
        checkpoint = callbacks.ModelCheckpoint(self.filepath+'model.h5',
            monitor='val_accuracy', mode='max', verbose=2, save_best_only=True)

        # train
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test']
        y_test = self.data['y_test']

        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
            steps_per_epoch=len(X_train) / 32, epochs=50, verbose=2,
            callbacks=[checkpoint], validation_data=(X_test, y_test))
        return model

    def evaluate(self, X_test, y_test):
        pred = []
        for i in range(NUM_MC):
            # get Monte Carlo samples by sampling network NUM_MC times
            # _, acc = self.model.evaluate(X_test, y_test)
            prediction = self.model.predict(X_test)[0][1]
            pred.append(prediction)
            # print(pred)
            # break
            # pred1.append(pred[1])
            # print(acc)
        sns.distplot(pred)
        import matplotlib.pyplot as plt
        plt.show()
        meanpred = np.mean(pred)
        stdpred = np.std(pred)
        print(meanpred)
        print(stdpred)


data = dataset.get_datadict()
datagen = dataset.get_datagen()

model = MCDropout('models/')
model.evaluate(data['X_test'], data['y_test'])
