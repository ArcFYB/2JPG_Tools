import os
import cv2
import numpy as np
from osgeo import gdal

#数据格式转化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def imgto8bit(img):
    img_nrm = normalization(img)
    img_8 = cv2.equalizeHist(np.uint8(255 * img_nrm))
    return img_8


def tif_jpg(rasterfile):
    in_ds = gdal.Open(rasterfile)  # 打开样本文件
    xsize = in_ds.RasterXSize  # 获取行列数
    ysize = in_ds.RasterYSize
    bands = in_ds.RasterCount
    B_band = in_ds.GetRasterBand(1)
    B= B_band.ReadAsArray(0, 0, xsize, ysize).astype(np.int16)
    G_band = in_ds.GetRasterBand(2)
    G = G_band.ReadAsArray(0, 0, xsize, ysize).astype(np.int16)
    R_band = in_ds.GetRasterBand(3)
    R = R_band.ReadAsArray(0, 0, xsize, ysize).astype(np.int16)
    R1 = imgto8bit(R)
    G1 = imgto8bit(G)
    B1 = imgto8bit(B)
    data2= cv2.merge([B1,G1,R1])
    return data2


# def tif_jpg(rasterfile):
#     in_ds = gdal.Open(rasterfile)  # 打开样本文件
#     xsize = in_ds.RasterXSize  # 获取行列数
#     ysize = in_ds.RasterYSize
#     geotransform = in_ds.GetGeoTransform()
#     bands = in_ds.RasterCount

#     B_band = in_ds.GetRasterBand(1)
#     B = B_band.ReadAsArray(0, 0, xsize, ysize)
#     G_band = in_ds.GetRasterBand(2)
#     G = G_band.ReadAsArray(0, 0, xsize, ysize)
#     R_band = in_ds.GetRasterBand(3)
#     R = R_band.ReadAsArray(0, 0, xsize, ysize)

#     R1 = ((R - np.min(R)) / (np.max(R)) - np.min(R)) * 256
#     R1 = cv2.equalizeHist(R1.astype(np.uint8))
#     G1 = ((G - np.min(G)) / (np.max(G)) - np.min(G)) * 256
#     G1 = cv2.equalizeHist(G1.astype(np.uint8))
#     B1 = ((B - np.min(B)) / (np.max(B)) - np.min(B)) * 256
#     B1 = cv2.equalizeHist(B1.astype(np.uint8))
#     data2 = cv2.merge([R1, G1, B1])

    # return data2
if __name__ == '__main__':
    path=r"C:\Users\Administrator\Desktop\gf\tif"
    classs = os.listdir(path)
    for idx, folder in enumerate(classs):
        if folder.endswith('tif') or folder.endswith('tiff') :
            ori_image = os.path.join(path, folder)
            print(ori_image)
            if folder.endswith('tiff'):
                result_name = os.path.basename(ori_image)[:-5]
            else:
                result_name = os.path.basename(ori_image)[:-4]
            # print(result_name)
            a = os.path.dirname(ori_image)
            out = a + "\\" + result_name + ".jpg"
            img=tif_jpg(ori_image)
            cv2.imencode('.jpg', img)[1].tofile(out)