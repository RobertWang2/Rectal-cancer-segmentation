'''
@date = '19/6/1'
@author = 'Robert Wang'
@email = '3255893782@qq.com'
'''
import os
import sys
import numpy as np
import random
import math
import tensorflow as tf
# from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Cropping2D,UpSampling2D,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras import backend as K
from skimage import io

#设置图像的维度顺序（‘tf’或‘th’）
K.set_image_data_format('channels_last')

# partB 接partA
image_rows = 192
image_cols = 128
IMG_WIDTH = 128
IMG_HEIGHT = 192
IMG_CHANNELS = 1
TOTAL = 2782 # 总共的训练数据
TOTAL_VAL = 152 # 总共的validation数据
# part1部分储存的数据文件
outputPath = './data_train/train_liver.h5' # 训练文件
val_outputPath = './data_train/val_liver.h5'
#checkpoint_path = 'model.ckpt'
BATCH_SIZE = 1 # 根据服务器的GPU显存进行调整
 
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 -dice_coef(y_true, y_pred)

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        print(target.shape)
        print(refer._keras_shape)
        cw = (target._keras_shape[2] - refer._keras_shape[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target._keras_shape[1] - refer._keras_shape[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)


def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    start_neurons = 16
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print("conv1 shape:", conv1.shape)
    conv1 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    print("conv1 shape:", conv1.shape)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)
    conv2 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    print("conv2 shape:", conv2.shape)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    conv3 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    print("conv3 shape:", conv3.shape)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(start_neurons * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = residual_block(conv5, start_neurons * 16)
    conv5 = residual_block(conv5, start_neurons * 16, True)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(start_neurons * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(start_neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = residual_block(conv6, start_neurons * 8)
    conv6 = residual_block(conv6, start_neurons * 8, True)

    up7 = Conv2D(start_neurons * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(start_neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = residual_block(conv7, start_neurons * 4)
    conv7 = residual_block(conv7, start_neurons * 4, True)

    up8 = Conv2D(start_neurons * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(start_neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = residual_block(conv8, start_neurons * 2)
    conv8 = residual_block(conv8, start_neurons * 2, True)

    up9 = Conv2D(start_neurons * 1, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(start_neurons * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = residual_block(conv9, start_neurons * 1)
    conv9 = residual_block(conv9, start_neurons * 1, True)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

#https://github.com/MrGiovanni/UNetPlusPlus
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout

import numpy as np
smooth = 1.
dropout_rate = 0.5
act = "relu"
########################################
# 2D Standard
########################################
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    # x = Conv2D(nb_filter, (kernel_size, kernel_size), name='conv' + stage + '_1', kernel_initializer='he_normal',
    #            padding='same', kernel_regularizer=l2(1e-4))(x)
    x = residual_block(x, nb_filter , True )
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    # x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    #
    # x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)


    return x

########################################

"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):

    nb_filter = [16,32,64,128,256]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
def unet():
    inpt = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    conv1 = Conv2d_BN(inpt, 16, (3, 3))
    conv1 = Conv2d_BN(conv1, 16, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2d_BN(pool1, 32, (3, 3))
    conv2 = Conv2d_BN(conv2, 32, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2d_BN(pool2, 64, (3, 3))
    conv3 = Conv2d_BN(conv3, 64, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = Conv2d_BN(pool3, 128, (3, 3))
    conv4 = Conv2d_BN(conv4, 128, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = Conv2d_BN(pool4, 256, (3, 3))
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2d_BN(conv5, 256, (3, 3))
    conv5 = Dropout(0.5)(conv5)

    convt1 = Conv2dT_BN(conv5, 128, (3, 3))
    concat1 = concatenate([conv4, convt1], axis=3)
    concat1 = Dropout(0.5)(concat1)
    conv6 = Conv2d_BN(concat1, 128, (3, 3))
    conv6 = Conv2d_BN(conv6, 128, (3, 3))

    convt2 = Conv2dT_BN(conv6, 64, (3, 3))
    concat2 = concatenate([conv3, convt2], axis=3)
    concat2 = Dropout(0.5)(concat2)
    conv7 = Conv2d_BN(concat2, 64, (3, 3))
    conv7 = Conv2d_BN(conv7, 64, (3, 3))

    convt3 = Conv2dT_BN(conv7, 32, (3, 3))
    concat3 = concatenate([conv2, convt3], axis=3)
    concat3 = Dropout(0.5)(concat3)
    conv8 = Conv2d_BN(concat3, 32, (3, 3))
    conv8 = Conv2d_BN(conv8, 32, (3, 3))

    convt4 = Conv2dT_BN(conv8, 16, (3, 3))
    concat4 = concatenate([conv1, convt4], axis=3)
    concat4 = Dropout(0.5)(concat4)
    conv9 = Conv2d_BN(concat4, 16, (3, 3))
    conv9 = Conv2d_BN(conv9, 16, (3, 3))
    conv9 = Dropout(0.5)(conv9)
    outpt = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)
    model = Model(inpt, outpt)

    return model

def get_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH , 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)

    ch, cw = get_crop_shape(conv4, up_conv5)

    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)

    up7 = concatenate([up_conv6, crop_conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)

    up8 = concatenate([up_conv7, crop_conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)


    up9 = concatenate([up_conv8, crop_conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
