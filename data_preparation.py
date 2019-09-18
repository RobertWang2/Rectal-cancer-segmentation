#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '19/6/1'
@author = 'Robert Wang'
@email = '3255893782@qq.com'
'''

from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import dicom
from sklearn.cluster import KMeans
from skimage.transform import resize
import glob
import os
from skimage import morphology
from skimage import measure
import numpy as np
import cv2

from skimage.io import imsave, imread
import SimpleITK as sitk
import pandas as pd
data_path = 'data/CT/1002/arterial phase/'
from PIL import Image
image_rows = 192
image_cols = 128
import matplotlib.pyplot as plt


def load_train_data():

    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train,imgs_mask_train


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_mask_test = np.load('imgs_mask_test.npy')
    return imgs_mask_test,imgs_mask_test


def get_pixels_hu_by_simpleitk(dicom_dir, intercept, slope):
    # 读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0

    # Convert to Hounsfield units (HU)

    if slope != 1:
        img_array = slope * img_array.astype(np.float64)
        img_array = img_array.astype(np.int16)

    img_array += np.int16(intercept)

    return np.array(img_array, dtype=np.int16)

def transform_ctdata(image, windowCenter, windowWidth, normal=False):
    """
    注意，这个函数的self.image一定得是float类型的，否则就无效！
    return: trucated image according to window center and window width
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


def laplaEigen(dataMat, k, t):
    m, n = np.shape(dataMat)
    W = np.mat(np.zeros([m, m]))
    D = np.mat(np.zeros([m, m]))
    for i in range(m):
        k_index = knn(dataMat[i, :], dataMat, k)
        for j in range(k):
            sqDiffVector = dataMat[i, :] - dataMat[k_index[j], :]
            sqDiffVector = np.array(sqDiffVector) ** 2
            sqDistances = sqDiffVector.sum()
            W[i, k_index[j]] = np.math.exp(-sqDistances / t)
            D[i, i] += W[i, k_index[j]]
    L = D - W
    Dinv = np.linalg.inv(D)
    X = np.dot(D.I, L)
    lamda, f = np.linalg.eig(X)


    return lamda, f


def knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = np.array(diffMat) ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()


    return sortedDistIndicies[0:k]

def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    #Generate a swiss roll dataset.
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    x = t * np.cos(t)
    y = 83 * np.random.rand(1, n_samples)
    z = t * np.sin(t)
    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t
def show(img):
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(img,'gray')
    # plt.imshow(img)
    #     # save_path= "结果/pic/直方图归一拉普拉斯/"+ name_train[5:9]+"_"+name_train[25:30] +".png"
    #     # plt.savefig(save_path)
    plt.show()
def normalization(img,norm = False):
    _min = img.min()
    _max = img.max()
    if _min!=_max:
        img-=_min
        img = img/(_max - _min)
        if norm == True:
            img*=255.0
    return img


def Image_preprocess():
    images_mask = glob.glob('data/*/arterial phase/*_mask.png')
    images = glob.glob('data/*/arterial phase/*.dcm')
    # images_mask = images_mask[:1000]
    # images = images[:1000]
    # cnt = 546*2
    # cnt = 860*2
    cnt = 1713
    imgs = np.ndarray((cnt, image_rows, image_cols))
    imgs_mask = np.ndarray((cnt, image_rows, image_cols))
    _names = np.ndarray(cnt,dtype=str)
    k0 = 0.67
    k1 = 0.501953125
    d = 64
    h = 96
    i = 0
    k = 0

    for (name_train,name_mask) in zip(images,images_mask):
        img_mask = cv2.imread(name_mask, 0)
        # img_train = name_train.pixel_array * name_train.RescaleSlope + name_train.RescaleIntercept
        img_array = sitk.ReadImage(name_train)
        img_train = np.squeeze(sitk.GetArrayFromImage(img_array))

        img_train[img_train < -1024] = -1024
        img_train = transform_ctdata(img_train, 45, 450)
        x = img_train
        # show(img_train)
        #plt.imshow(img_train,'gray')
        # plt.show()
        x2 = k0 * img_train.shape[0] + h
        x1 = k0 * img_train.shape[0] - h
        y1 = k1 * img_train.shape[1] - d
        y2 = k1 * img_train.shape[1] + d
        x0 = k0 * img_train.shape[0]
        y0 = k1 * img_train.shape[1]
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        img_train = img_train[x1:x2, y1:y2]
        img_train = img_train.astype('float32')
        img_mask = img_mask.astype('float32')
        img_mask = img_mask[x1:x2, y1:y2]
        # img_mask = img_mask.astype('int32')
        # temp = img_mask*img_train
        # temp[temp<0]=0
        # kernel = np.ones((3, 3), np.float32)
        # img_train = cv2.morphologyEx(img_train, cv2.MORPH_OPEN, kernel)
        # plt.imshow(temp,'gray')
        # plt.show()
        # img_train[img_train < -956]=0
        # img_train[img_train > 386] = 0
        # img_train = transform_ctdata(img_train,671,-285)
        # 强度归一化
        # img_train = normalization(img_train,True)
        # 窗位窗宽调整
        # img_train = transform_ctdata(img_train, 45, 450)
        # show(img_train)
        # show(img_mask)

        # 开运算
        kernel = np.ones((3, 3), np.float32)
        img_train = cv2.morphologyEx(img_train, cv2.MORPH_OPEN, kernel)


        # 直方图均衡化

        img_train = np.uint8(cv2.normalize(img_train, None, 0, 255, cv2.NORM_MINMAX))
        img_train = cv2.equalizeHist(img_train)
        # show(img_train)

        # 归一化
        sc = StandardScaler()
        img_train = sc.fit_transform(img_train)
        show(img_train)

        img_mask[img_mask > 1] = 1
        # plt.imshow(img_train, 'gray')
        # plt.show()

        #拉普拉斯
        # gray_lap = cv2.Laplacian(img_train, cv2.CV_16S, ksize=3)
        # img_train = cv2.convertScaleAbs(gray_lap)

        # 聚类
        # kmeans = KMeans(n_clusters=2).fit(np.reshape(img_train, [np.prod(img_train.shape), 1]))
        # centers = sorted(kmeans.cluster_centers_.flatten())
        # threshold = np.mean(centers)
        # threshold = (img_train.min()+img_train.max())/2
        # img_train = np.where(img_train >= threshold, 1.0, 0.0)
        # 拉普拉斯
        # gray_lap = cv2.Laplacian(img_train, cv2.CV_16S, ksize=3)
        # img_train = cv2.convertScaleAbs(gray_lap)

        # for j in range(0,image_cols) :
        #   if img_mask[0][j] == 1:
        #     print(i,1)
        #     break
        #   elif img_mask[-1][j] == 1:
        #     print(i,2)
        #     break

        # img_mask /= 255.

        if img_mask.max() >= 1:
            # if img_mask.max() >= 1 :
            # show(img_train, name_train)
            k += 1
            imgs[i] = img_train
            imgs_mask[i] = img_mask
            _names[i] = name_mask
            print('convert images {} in {} {}'.format(i+1, len(images),name_train))
            # show(x)
            i += 1
        elif k:
            imgs[i] = img_train
            imgs_mask[i] = img_mask
            print('convert images {} in {}'.format(i + 1, len(images)))
            # show(x)
            i += 1
            k -= 1
        # for j in range(0,128) :
        # if i > 19 :
        #     plt.imshow(img_train, 'gray')
        #     plt.show()

    # print('k = ',k)
    print('Loading done.')
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')
    # name_mask =  name_mask[5:9]+"_"+name_mask[25:30]


def test_data():
    images_mask = glob.glob('data/1001/arterial phase/*_mask.png')
    images = glob.glob('data/1001/arterial phase/*.dcm')
    cnt = len(images)
    imgs = np.ndarray((cnt, image_rows, image_cols))
    imgs_mask = np.ndarray((cnt, image_rows, image_cols))
    _names = np.ndarray(cnt, dtype=str)
    k0 = 0.67
    k1 = 0.501953125
    d = 64
    h = 96
    i = 0
    k = 0

    for (name_train, name_mask) in zip(images, images_mask):
        img_mask = cv2.imread(name_mask, 0)
        img_array = sitk.ReadImage(name_train)
        img_train = np.squeeze(sitk.GetArrayFromImage(img_array))
        img_train[img_train < -1024] = -1024
        img_train = transform_ctdata(img_train, 45, 450)
        # show(img_train)
        # plt.imshow(img_train,'gray')
        # plt.show()
        x2 = k0 * img_train.shape[0] + h
        x1 = k0 * img_train.shape[0] - h
        y1 = k1 * img_train.shape[1] - d
        y2 = k1 * img_train.shape[1] + d
        x0 = k0 * img_train.shape[0]
        y0 = k1 * img_train.shape[1]
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        img_train = img_train[x1:x2, y1:y2]
        img_train = img_train.astype('float32')
        img_mask = img_mask.astype('float32')
        img_mask = img_mask[x1:x2, y1:y2]

        # 开运算
        kernel = np.ones((3, 3), np.float32)
        img_train = cv2.morphologyEx(img_train, cv2.MORPH_OPEN, kernel)

        # 直方图均衡化

        img_train = np.uint8(cv2.normalize(img_train, None, 0, 255, cv2.NORM_MINMAX))
        img_train = cv2.equalizeHist(img_train)
        # show(img_train)

        # 归一化
        sc = StandardScaler()
        img_train = sc.fit_transform(img_train)

        img_mask[img_mask > 1] = 1
        imgs[i] = img_train
        imgs_mask[i] = img_mask
        i+=1
        print('convert images {} in {} {}'.format(i + 1, len(images), name_train))

    print('Loading done.')
    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_mask)
    print('Saving to .npy files done.')


if __name__ == '__main__':
    Image_preprocess()
    # test_data()
    # 从门脉期图像中得到的分类结果