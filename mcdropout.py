import keras
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

NUM_CLASSES = 2
NUM_MC = 100

class MCDropout:
    def __init__(self, filepath, data=None):
        try:
            self.model = models.load_model(filepath+'/model.h5')
        except:
            self.data = data
            self.model = self.train()

    def train(self):
        inp = Input(shape)
        x = Conv2D(16, kernel_size=(3, 3), activation=act)(inp)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernel_size=(3, 3), activation=act)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(x, p=0.5, training=True)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(x, p=0.5)
        out = Dense(NUM_CLASSES, activation='softmax')(x)

        model = Model(inputs=inp, outputs=out)
        optimizer = keras.optimizers.Adam(lr=1e-3 beta_1=0.9, beta_2=0.999,
            amsgrad=False)

        model.compile(loss=keras.losses.binary_crossentropy,
            optimizer=optimizer, metrics=['accuracy'])

        checkpoint = callbacks.ModelCheckpoint('/model.h5',
            monitor='acc', mode='val_acc', verbose=2,
            save_best_only=True)

        model.fit(self.data['X_train'], self.data['y_train'],
            validation_data=[self.data['X_val'], self.data['y_val']],
            epochs=50, batch_size=64, verbose=2,
            callbacks=[checkpoint, csv_logger])

        return model

        def evaluate(self):
            for i in range(NUM_MC):
                model.predict(self.data['X_test'])
