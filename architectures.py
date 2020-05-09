from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ZeroPadding2D
from keras.applications import VGG16

def CNN(NUM_CLASSES):
    input = Input(shape=(224, 224 ,NUM_CLASSES))
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
    return model

def VGG16(NUM_CLASSES):
    # pretrained VGG16
    input =  = Input(shape=(224, 224, NUM_CLASSES))
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=input,
        pooling="max"
    )
    x = Flatten()(base_model.outputs)
    x = Dense(1024, activation='relu')(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def AlexNet(NUM_CLASSES):
    # https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-
    # convolutional-neural-networks.pdf
    input = Input(shape=(224, 224 ,3))
        # first layer
    x = Conv2D(96, kernel_size=(11, 11), padding = 'valid', strides=(4,4), activation='relu')(input)
    x = MaxPooling2D(pool_size=(3, 3), strides = (2,2))(x)
        # second layer
    x = ZeroPadding2D((2,2))(x)
    x = Conv2D(256, kernel_size=(5, 5), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = (2,2), padding= 'valid')(x)
        # third layer
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(384, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
        # fourth layer
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(384, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
        # fifth layer
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, kernel_size=(3, 3), strides = (1, 1), padding = 'same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides = (2, 2), padding = 'valid')(x)
        # fully connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    return model
