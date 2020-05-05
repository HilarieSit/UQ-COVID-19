# import dataset
# X, y = dataset.return_Xy()

x = tf.placeholder(tf.float32, shape=[None, 244, 244, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])


# code from Tensorflow Probability Authors
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py
kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p)
                            tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

model = tf.keras.Sequential([
      tfp.layers.Convolution2DFlipout(32, kernel_size=5, padding="SAME", kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2],  padding="SAME"),
      tfp.layers.Convolution2DFlipout(64, kernel_size=5, padding="SAME",  kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
      tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
      tf.keras.layers.Flatten(),
      tfp.layers..DenseFlipout(10, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.relu),
      tfp.layers.DenseFlipout(2, kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.softmax),
optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
model.compile(optimizer, loss='binary_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)


# Tensorflow's Dataset pipeline with reinitializable iterator: https://www.tensorflow.org/guide/datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(100).batch(bs).prefetch(1).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(n_test).prefetch(1)
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):
        sess.run(train_init_op, feed_dict = {x : X_train, y: y_train})
        loss_total = 0
        for j in range(n_batch):
            _, train_loss = sess.run([model.train_op, model.loss])
            loss_total += train_loss
