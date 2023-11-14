'''

将卫星图像（tiff）格式裁剪为固定大小的图像块（如256x256，128x128等），用于后续的深度学习训练

'''
import sys
import cv2
from osgeo import gdal
print('gdal version: ', gdal.__version__)

import random
import os
import numpy as np


def auto_level_threshold(hist, ratio=0.0):
    len = hist.shape[0]
    print('len: ', len)
    sum = np.sum(hist)
    print(sum)
    pixel_num = sum*ratio
    
    s = 0
    l = 0
    while s<pixel_num:
        s = s + hist[l]
        l = l + 1

    s = 0
    r = len-1
    while s<pixel_num:
        s = s + hist[r]
        r = r - 1
    #print('auto level threshold:', l, r)
    return l,r

def generateGaojingSamples(file):
    #print(file)
    #print(file.split('/'))
    title= file.split('/')[-1]
    title=title[:-5]
    print(title)

    dataset = gdal.Open(file)
    arr = dataset.ReadAsArray()          #(3, ht, wd)
    arr = arr.transpose(1, 2, 0)         #(ht, wd, 3)
    print(arr.shape, arr.dtype)
    ht = arr.shape[0]
    wd = arr.shape[1]
    nc = arr.shape[2]

    for i in range(nc):
        single = arr[:,:,i]
        # hist = np.bincount(single.ravel())
        # lv, rv = auto_level_threshold(hist, ratio=0.005)
        # minv = lv #tiff_buf.min()
        # maxv = rv #tiff_buf.max()
        minv = np.min(single)
        maxv = np.max(single)
        print(minv,maxv)
        arr[:, :, i] = (single - minv)/(maxv-minv)*255
        # index = np.where(arr[:, :, i] > 255)
        # arr[:, :, i][index] = 255
        # index = np.where(arr[:, :, i] < 0)
        # arr[:, :, i][index] = 0
        #arr[:,:,i] = cv2.equalizeHist(single)

    arr = arr.astype(np.uint8)
    print(arr.dtype)
    
    #直方图均衡化
    for i in range(nc):
        single = arr[:,:,i]
        arr[:,:,i] = cv2.equalizeHist(single)
    cv2.imwrite('/home/xdh/equalize.jpg', arr[:,:,0:3])

    out_path = '/home/xdh/data/super-resolution/GJ/train'
    image_size = 256
    for i in range(10000):
        l = int( random.random()*(wd-image_size-1) )
        t = int( random.random()*(ht-image_size-1) )
        #print(l, t)
        sample_patch = arr[t:t+image_size,l:l+image_size,0:3]
        out_file = os.path.join(out_path,title+"_"+str(i)+'.jpg')
        print(out_file, sample_patch.shape)
        cv2.imwrite(out_file, sample_patch)

if __name__=='__main__':
    print('generate image patch ... ')
    file = sys.argv[1]
    generateGaojingSamples(file)
    






