#!/usr/bin/env python3.5
# coding=utf-8
'''
@date = '19/6/1'
@author = 'Robert Wang'
@email = '3255893782@qq.com'
'''
from __future__ import print_function

import numpy as np
import pandas as pd
import six
import glob
from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize



from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.regularizers import l2

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

import os
from skimage.io import imsave, imread
from skimage.transform import resize
import os
import sys
import random

import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

# import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.core import regularizers

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
from skimage.io import imsave
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from data_preparation import load_train_data, load_test_data,test_data,Image_preprocess
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
import cv2
img_rows = 192
img_cols = 128
smooth = 1.
# CUDA_VISIBLE_DEVICES="0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Reshape, Activation, core, Permute
import model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def IOU(y_true, y_pred, eps=1e-6):
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def BatchActivate(x,activate = 'relu'):
    x = BatchNormalization()(x)
    x = Activation(activate)(x)
    return x
import keras.initializers as init
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=False,a='relu'):
    x = Conv2D(filters, size, strides=strides, padding=padding,kernel_initializer=init.random_normal())(x)
    if activation == True:
        x = BatchActivate(x,a)
    return x

def residual_block_b(blockInput, num_filters=16, batch_activate = False,a='relu'):
    x = BatchActivate(blockInput,a)
    # x = blockInput
    # x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3,3), activation=True,a=a)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x,a)
    return x

def residual_block_c(blockInput, num_filters=16, batch_activate = False,a='relu'):
    x = BatchActivate(blockInput,a)
    x = convolution_block(x, num_filters, (3,3))
    # blockInput = Add()([x, blockInput])
    x = convolution_block(blockInput, num_filters, (3,3), activation=True)
    # x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x,a)
    return x


def build_model(input_layer, start_neurons, DropoutRatio=0.5,regurate=0.000001):
    # 101 -> 50

    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(input_layer)
    # conv1 = residual_block_b(conv1, start_neurons * 1)
    conv1 = residual_block_b(conv1, start_neurons * 1, True,a='selu')
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio )(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(pool1)
    # conv2 = residual_block_b(conv2, start_neurons * 2)
    conv2 = residual_block_b(conv2, start_neurons * 2, True,a='selu')
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(pool2)
    # conv3 = residual_block_b(conv3, start_neurons * 4)
    conv3 = residual_block_b(conv3, start_neurons * 4, True,a='selu')
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(pool3)
    # conv4 = residual_block_b(conv4, start_neurons * 8)
    conv4 = residual_block_b(conv4, start_neurons * 8, True,a='selu')
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(pool4)
    # convm = residual_block_b(convm, start_neurons * 16)
    convm = residual_block_b(convm, start_neurons * 16, True,a='selu')

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init.uniform(),kernel_regularizer=regularizers.l2(regurate))(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(uconv4)
    # uconv4 = residual_block_b(uconv4, start_neurons * 8)
    uconv4 = residual_block_b(uconv4, start_neurons * 8, True,a='selu')

    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init.uniform(),kernel_regularizer=regularizers.l2(regurate))(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(uconv3)
    # uconv3 = residual_block_b(uconv3, start_neurons * 4)
    uconv3 = residual_block_b(uconv3, start_neurons * 4, True,a='selu')

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init.uniform(),kernel_regularizer=regularizers.l2(regurate))(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(uconv2)
    # uconv2 = residual_block_b(uconv2, start_neurons * 2)
    uconv2 = residual_block_b(uconv2, start_neurons * 2, True,a='selu')

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same",kernel_initializer=init.uniform(),kernel_regularizer=regularizers.l2(regurate))(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_regularizer=regularizers.l2(regurate))(uconv1)
    # uconv1 = residual_block_b(uconv1, start_neurons * 1)
    uconv1 = residual_block_b(uconv1, start_neurons * 1, True,a='selu')

    uconv1 = Dropout(DropoutRatio)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None,kernel_regularizer=regularizers.l2(regurate))(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer


def DataAugmentation(images,masks):
    from keras.preprocessing.image import ImageDataGenerator

    data_gen_args = dict(rotation_range=3,
                         width_shift_range=0.01,
                         height_shift_range=0.01,
                         shear_range=0.01,
                         zoom_range=0.01,
                         fill_mode='nearest')
    seed = 2019
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    image_generator = image_datagen.flow(images, seed=seed)
    mask_generator = mask_datagen.flow(masks, seed=seed)
    train_generator = zip(image_generator, mask_generator)
    print('build ImageDataGenerator finished.')
    return train_generator

import keras
class MyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.dice_coef = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_dice_coef = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.dice_coef['batch'].append(logs.get('dice_coef'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_dice_coef['batch'].append(logs.get('val_dice_coef'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.dice_coef['epoch'].append(logs.get('dice_coef'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_dice_coef['epoch'].append(logs.get('val_dice_coef'))

    def loss_plot(self, loss_type):
        import pandas as pd

        iters = range(len(self.losses[loss_type]))
        dice = pd.DataFrame({'train dice':self.dice_coef[loss_type],'val dice':self.val_dice_coef[loss_type]})
        dice.to_excel('./结果/运行记录/unet_dice.xlsx',index=False)

        loss = pd.DataFrame({'train loss': self.losses[loss_type], 'val loss': self.val_loss[loss_type]})
        loss.to_excel('./结果/运行记录/unet_loss.xlsx', index=False)
        plt.figure()
        # dice
        plt.plot(iters, self.dice_coef[loss_type], 'r', label='train dice')
        # loss
        # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_dice
            plt.plot(iters, self.val_dice_coef[loss_type], 'b', label='val dice')
            # val_loss
            # plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('dice Similarity')
        plt.legend(loc="upper right")
        plt.show()


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols))
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows ,img_cols), preserve_range=True)
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train():
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    input_layer = Input((img_rows, img_cols,1))
    import models
    model1 = models.get_unet()
    output_layer = build_model(input_layer, 16)
    # model1 = Model(input_layer,output_layer)

    weight_path = 'model/unet_weights.h5'
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)
    history = MyHistory()
    reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=20, verbose=1, mode='auto', epsilon=0.0001,
                                       cooldown=5, min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=15)  # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat,history]
    c = optimizers.adam(lr=0.001)
    from keras.losses import binary_crossentropy
    model1.compile(loss=dice_coef_loss, optimizer=c, metrics=[dice_coef])

    train_generator = DataAugmentation(images=imgs_train,masks=imgs_mask_train)


    # model1.fit(imgs_train, imgs_mask_train, batch_size=16, nb_epoch=200, verbose=1, shuffle=True,
    #           validation_split=0.1,callbacks=[history] )
    model1.fit_generator(train_generator,
                              epochs = 200, validation_data = (imgs_train[-400:],imgs_mask_train[-400:]),
                              verbose = 2, steps_per_epoch=imgs_train.shape[0] // 16
                              ,callbacks = [history],shuffle=True)

    model1.save('model/unet_weights.h5')

    print('Predicting masks on test data...')

    # imgs_mask_test = model1.predict(imgs_train, verbose=-1)


    print('success')
    # np.save('imgs_mask_test.npy', imgs_mask_test)

    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    imgs_test, imgs_mask_test = load_test_data()
    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)
    pred_dir = 'preds'
    imgs_mask_test = model1.predict(imgs_test)
    for i, image in enumerate(imgs_mask_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        # ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
    history.loss_plot('epoch')


def predict():
    input_layer = Input((img_rows, img_cols, 1))
    output_layer = build_model(input_layer, 16, 0.5)
    model1 = Model(input_layer, output_layer)
    # model1 = model.get_unet()
    weight_path = 'model/unet_weights.h5'
    model1.load_weights(weight_path)
    imgs_test, imgs_mask_test = load_train_data()
    imgs_test = preprocess(imgs_test[:20])
    imgs_mask_test = preprocess(imgs_mask_test[:20])
    pred_dir = 'preds'
    imgs_mask_test = model1.predict(imgs_test)
    for cnt,image in enumerate(imgs_mask_test):

        image = (image[:, :, 0] * 255.).astype(np.uint8)
        ret,image = cv2.threshold(image,10,255,cv2.THRESH_BINARY)
        # 图像还原
        mask = imread('mask.png')
        mask = mask[..., np.newaxis]
        # mask = mask.astype('float32')
        k0 = 0.69
        k1 = 0.501953125
        d = 64
        x2 = k0 * mask.shape[0] + d
        x1 = k0 * mask.shape[0] - d
        y1 = k1 * mask.shape[1] - d
        y2 = k1 * mask.shape[1] + d
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        x = mask[x1:x2, y1:y2, :]
        i = j = 0
        for i in np.arange(x1, x2):
            for j in np.arange(y1, y2):
                mask[i, j] += image[i - (x1), j - (y1)]
        image = mask
        image = image[:,:,0]
        imsave(os.path.join(pred_dir, str(cnt) + '_pred.png'), image)

    print('predict done!')

if __name__ == '__main__':

    import data_preparation

    train()
    # predict()
