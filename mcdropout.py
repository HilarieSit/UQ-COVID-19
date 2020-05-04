import keras
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import dataset
import seaborn as sns

NUM_CLASSES = 2         # number of classes
NUM_MC = 100            # number of monte carlo samples

class MCDropout:
    def __init__(self, filepath, datagen=None, data=None):
        try:
            self.model = models.load_model(filepath+'/model.h5')
        except:
            self.data = data
            self.model = self.train(datagen)

    def train(self, datagen):
        # architecture
        input = Input(shape=(244,244,3))
        # default is channels last
        x = Conv2D(16, kernel_size=(3, 3), activation=act)(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x, training=True)
        x = Conv2D(32, kernel_size=(3, 3), activation=act)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x, training=True)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=input, outputs=output)

        # optimizer/loss
        optimizer = keras.optimizers.Adam(lr=1e-3 beta_1=0.9, beta_2=0.999,
            amsgrad=False)
        model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=optimizer, metrics=['accuracy'])

        # train
        model.fit_generator(datagen.flow(self.data['X_train'], self.data['y_train'], batch_size=64),
            steps_per_epoch=len(x_train) / 32, epochs=50, verbose=2,
            callbacks=[checkpoint])
        return model

    def evaluate(self):
        pred = []
        for i in range(NUM_MC):
            # get Monte Carlo samples by sampling network NUM_MC times
            pred.append(self.model.predict(self.data['X_test'])
        meanpred = pred.mean()
        stdpred = pred.std()

MCDropout('models/', datagen=dataset.get_datagen(), data=dataset.get_datadict())
