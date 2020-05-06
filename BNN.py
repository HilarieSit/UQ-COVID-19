import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import dataset

class BNN:
    def __init__(self, datagen=None, data=None):
        self.train(datagen, data)

    def train(self, datagen, data):
        # train
        X_train = data['X_train']
        n_train_ex = len(X_train)
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        # code from Tensorflow Probability Authors
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p),
                                    tf.cast(n_train_ex, dtype=tf.float32))
        model = tf.keras.Sequential([
              tfp.layers.Convolution2DFlipout(32, kernel_size=5, padding="SAME", kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
              tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2],  padding="SAME"),
              tfp.layers.Convolution2DFlipout(64, kernel_size=5, padding="SAME",  kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
              tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
              tf.keras.layers.Flatten(),
              tfp.layers.DenseFlipout(10, kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.relu),
              tfp.layers.DenseFlipout(2, kernel_divergence_fn=kl_divergence_function,
                  activation=tf.nn.softmax)
        ])

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer, loss='binary_crossentropy',
                        metrics=['accuracy'], experimental_run_tf_function=False)

        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
            steps_per_epoch=len(X_train) / 32, epochs=50, verbose=2,
            validation_data=(X_test, y_test))

data = dataset.get_datadict()
datagen = dataset.get_datagen()
BNN(datagen=datagen, data=data)
