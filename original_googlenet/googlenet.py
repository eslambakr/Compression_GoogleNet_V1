# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
from keras import backend as K

keras.backend.set_image_data_format('channels_first')

from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, concatenate
from keras.datasets import cifar10
from keras.regularizers import l2
from keras.models import Model

from sklearn.metrics import log_loss

from original_googlenet.custom_layers.googlenet_custom_layers import LRN2D, PoolHelper

from original_googlenet.load_cifar10 import load_cifar10_data


def googlenet_model(img_rows, img_cols, channel=1, num_classes=None, trainable=True):
    """
    GoogLeNet a.k.a. Inception v1 for Keras

    Model Schema is based on 
    https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14

    ImageNet Pretrained Weights 
    https://drive.google.com/open?id=0B319laiAPjU3RE1maU9MMlh2dnc

    Blog Post: 
    http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """

    input = Input(shape=(channel, img_rows, img_cols))
    conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1/7x7_s2',
                          kernel_regularizer=l2(0.0002), trainable=trainable)(input)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1), trainable=trainable)(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1/3x3_s2',
                                trainable=trainable)(pool1_helper)
    pool1_norm1 = LRN2D(name='LRN_1')(pool1_3x3_s2)
    conv2_3x3_reduce = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2/3x3_reduce',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(pool1_norm1)

    conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002),
                       trainable=trainable)(conv2_3x3_reduce)

    conv2_norm2 = LRN2D(name='LRN_2')(conv2_3x3)

    conv2_zero_pad = ZeroPadding2D(padding=(1, 1), trainable=trainable)(conv2_norm2)

    pool2_helper = PoolHelper()(conv2_zero_pad)

    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2/3x3_s2',
                                trainable=trainable)(pool2_helper)

    inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3a/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(pool2_3x3_s2)

    inception_3a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu', name='inception_3a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool2_3x3_s2)

    inception_3a_3x3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='inception_3a/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_3x3_reduce)

    inception_3a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu', name='inception_3a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool2_3x3_s2)

    inception_3a_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu', name='inception_3a/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_5x5_reduce)

    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3a/pool',
                                     trainable=trainable)(pool2_3x3_s2)

    inception_3a_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_3a/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_pool)

    inception_3a_output = concatenate([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                      axis=1)

    inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_output)

    inception_3b_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_3b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_output)

    inception_3b_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu', name='inception_3b/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3b_3x3_reduce)

    inception_3b_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_3b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3a_output)

    inception_3b_5x5 = Conv2D(96, (5, 5), padding='same', activation='relu', name='inception_3b/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3b_5x5_reduce)

    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_3b/pool',
                                     trainable=trainable)(inception_3a_output)

    inception_3b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_3b/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_3b_pool)

    inception_3b_output = concatenate([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                      axis=1)

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1), trainable=trainable)(inception_3b_output)

    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool3/3x3_s2',
                                trainable=trainable)(pool3_helper)

    inception_4a_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu', name='inception_4a/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(pool3_3x3_s2)

    inception_4a_3x3_reduce = Conv2D(96, (1, 1), padding='same', activation='relu', name='inception_4a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool3_3x3_s2)

    inception_4a_3x3 = Conv2D(208, (3, 3), padding='same', activation='relu', name='inception_4a/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4a_3x3_reduce)

    inception_4a_5x5_reduce = Conv2D(16, (1, 1), padding='same', activation='relu', name='inception_4a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool3_3x3_s2)

    inception_4a_5x5 = Conv2D(48, (5, 5), padding='same', activation='relu', name='inception_4a/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4a/pool',
                                     trainable=trainable)(pool3_3x3_s2)

    inception_4a_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_4a/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4a_pool)

    inception_4a_output = concatenate([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                      axis=1)

    loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool', trainable=trainable)(
        inception_4a_output)

    loss1_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss1/conv',
                        kernel_regularizer=l2(0.0002), trainable=trainable)(loss1_ave_pool)

    loss1_flat = Flatten()(loss1_conv)

    loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002), trainable=trainable)(
        loss1_flat)

    loss1_drop_fc = Dropout(0.7, trainable=trainable)(loss1_fc)

    loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002), trainable=trainable)(
        loss1_drop_fc)

    loss1_classifier_act = Activation('softmax', name='head1')(loss1_classifier)

    inception_4b_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='inception_4b/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4a_output)

    inception_4b_3x3_reduce = Conv2D(112, (1, 1), padding='same', activation='relu', name='inception_4b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4a_output)

    inception_4b_3x3 = Conv2D(224, (3, 3), padding='same', activation='relu', name='inception_4b/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_3x3_reduce)

    inception_4b_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu', name='inception_4b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4a_output)

    inception_4b_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4b/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_5x5_reduce)

    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4b/pool',
                                     trainable=trainable)(inception_4a_output)

    inception_4b_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_4b/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_pool)

    inception_4b_output = concatenate([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                      axis=1)

    inception_4c_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_4c/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_output)

    inception_4c_3x3_reduce = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_4c/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_output)

    inception_4c_3x3 = Conv2D(256, (3, 3), padding='same', activation='relu', name='inception_4c/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_3x3_reduce)

    inception_4c_5x5_reduce = Conv2D(24, (1, 1), padding='same', activation='relu', name='inception_4c/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4b_output)

    inception_4c_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4c/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_5x5_reduce)

    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4c/pool',
                                     trainable=trainable)(inception_4b_output)

    inception_4c_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_4c/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_pool)

    inception_4c_output = concatenate([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                      axis=1)

    inception_4d_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='inception_4d/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_output)

    inception_4d_3x3_reduce = Conv2D(144, (1, 1), padding='same', activation='relu', name='inception_4d/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_output)

    inception_4d_3x3 = Conv2D(288, (3, 3), padding='same', activation='relu', name='inception_4d/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_3x3_reduce)

    inception_4d_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_4d/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4c_output)

    inception_4d_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', name='inception_4d/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_5x5_reduce)

    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4d/pool',
                                     trainable=trainable)(inception_4c_output)

    inception_4d_pool_proj = Conv2D(64, (1, 1), padding='same', activation='relu', name='inception_4d/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_pool)

    inception_4d_output = concatenate([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                      axis=1)

    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool', trainable=trainable)(
        inception_4d_output)

    loss2_conv = Conv2D(128, (1, 1), padding='same', activation='relu', name='loss2/conv',
                        kernel_regularizer=l2(0.0002), trainable=trainable)(loss2_ave_pool)

    loss2_flat = Flatten()(loss2_conv)

    loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002), trainable=trainable)(
        loss2_flat)

    loss2_drop_fc = Dropout(0.7, trainable=trainable)(loss2_fc)

    loss2_classifier = Dense(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002), trainable=trainable)(
        loss2_drop_fc)

    loss2_classifier_act = Activation('softmax', name='head2')(loss2_classifier)

    inception_4e_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_4e/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_output)

    inception_4e_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu', name='inception_4e/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_output)

    inception_4e_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_4e/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4e_3x3_reduce)

    inception_4e_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_4e/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4d_output)

    inception_4e_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_4e/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4e_5x5_reduce)

    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_4e/pool',
                                     trainable=trainable)(inception_4d_output)

    inception_4e_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_4e/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_4e_pool)

    inception_4e_output = concatenate([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                      axis=1)

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1), trainable=trainable)(inception_4e_output)

    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool4/3x3_s2',
                                trainable=trainable)(pool4_helper)

    inception_5a_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu', name='inception_5a/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(pool4_3x3_s2)

    inception_5a_3x3_reduce = Conv2D(160, (1, 1), padding='same', activation='relu', name='inception_5a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool4_3x3_s2)

    inception_5a_3x3 = Conv2D(320, (3, 3), padding='same', activation='relu', name='inception_5a/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_3x3_reduce)

    inception_5a_5x5_reduce = Conv2D(32, (1, 1), padding='same', activation='relu', name='inception_5a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(pool4_3x3_s2)

    inception_5a_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5a/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_5x5_reduce)

    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5a/pool',
                                     trainable=trainable)(pool4_3x3_s2)

    inception_5a_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_5a/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_pool)

    inception_5a_output = concatenate([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                      axis=1)

    inception_5b_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu', name='inception_5b/1x1',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_output)

    inception_5b_3x3_reduce = Conv2D(192, (1, 1), padding='same', activation='relu', name='inception_5b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_output)

    inception_5b_3x3 = Conv2D(384, (3, 3), padding='same', activation='relu', name='inception_5b/3x3',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5b_3x3_reduce)

    inception_5b_5x5_reduce = Conv2D(48, (1, 1), padding='same', activation='relu', name='inception_5b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5a_output)

    inception_5b_5x5 = Conv2D(128, (5, 5), padding='same', activation='relu', name='inception_5b/5x5',
                              kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5b_5x5_reduce)

    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='inception_5b/pool',
                                     trainable=trainable)(inception_5a_output)

    inception_5b_pool_proj = Conv2D(128, (1, 1), padding='same', activation='relu', name='inception_5b/pool_proj',
                                    kernel_regularizer=l2(0.0002), trainable=trainable)(inception_5b_pool)

    inception_5b_output = concatenate([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                      axis=1)

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2', trainable=trainable)(
        inception_5b_output)

    loss3_flat = Flatten()(pool5_7x7_s1)

    pool5_drop_7x7_s1 = Dropout(0.4, trainable=trainable)(loss3_flat)

    loss3_classifier = Dense(1000, name='loss3/classifier', kernel_regularizer=l2(0.0002), trainable=trainable)(
        pool5_drop_7x7_s1)

    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    # Create model
    model = Model(input=input, output=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    # Load ImageNet pre-trained data 
    model.load_weights('imagenet_models/googlenet_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    loss3_classifier_statefarm = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002),
                                       trainable=trainable)(
        pool5_drop_7x7_s1)
    loss3_classifier_act_statefarm = Activation('softmax', name='prob')(loss3_classifier_statefarm)
    loss2_classifier_statefarm = Dense(num_classes, name='loss2/classifier', kernel_regularizer=l2(0.0002),
                                       trainable=trainable)(loss2_drop_fc)
    loss2_classifier_act_statefarm = Activation('softmax')(loss2_classifier_statefarm)
    loss1_classifier_statefarm = Dense(num_classes, name='loss1/classifier', kernel_regularizer=l2(0.0002),
                                       trainable=trainable)(loss1_drop_fc)
    loss1_classifier_act_statefarm = Activation('softmax')(loss1_classifier_statefarm)

    # Create another model with our customized softmax
    model = Model(input=input, output=[loss1_classifier_act_statefarm, loss2_classifier_act_statefarm,
                                       loss3_classifier_act_statefarm])

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 1

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = googlenet_model(img_rows, img_cols, channel, num_classes)
    model.summary()
    # Start Fine-tuning. 
    # Notice that googlenet takes 3 sets of labels for outputs, one for each auxillary classifier
    model.fit(X_train, [Y_train, Y_train, Y_train],
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, [Y_valid, Y_valid, Y_valid]),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Combine 3 set of outputs using averaging
    predictions_valid = sum(predictions_valid) / len(predictions_valid)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print("score = ", score)
