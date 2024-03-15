'''

批量将文件夹中的.tif和.tiff转换为.jpg
进行对比度增强

'''
import os
import cv2
import numpy as np
from osgeo import gdal
from PIL import Image, ImageEnhance

#数据格式转化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
def imgto8bit(img):
    img_nrm = normalization(img)
    img_8 = np.uint8(255 * img_nrm)
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
    data2= cv2.merge([R1,G1,B1])
    return data2

def enhance_image(input_path, output_path, contrast_factor=1.5, color_factor=1.5):
    with Image.open(input_path) as img:
        # 对比度增强
        contrast = ImageEnhance.Contrast(img)
        img_contrast = contrast.enhance(contrast_factor)

        # 色彩增强
        color = ImageEnhance.Color(img_contrast)
        img_enhanced = color.enhance(color_factor)

        # 保存增强后的图片
        img_enhanced.save(output_path)

if __name__ == '__main__':
    path=r"C:\Users\Administrator\Desktop\tif_dataset\1"
    classs = os.listdir(path)
    for idx, folder in enumerate(classs):
        if folder.endswith('tif') or folder.endswith('tiff') :
            ori_image = os.path.join(path, folder)
            print(ori_image)
            if folder.endswith('tiff'):
                result_name = os.path.basename(ori_image)[:-5]
            else:
                result_name = os.path.basename(ori_image)[:-4]
            a = os.path.dirname(ori_image)
            out = a + "\\" + result_name + ".jpg"
            img=tif_jpg(ori_image)
            cv2.imencode('.jpg', img)[1].tofile(out)
            # Operation : enhance_image
            input_image_path = out
            output_image_path = a + "\\" + result_name + "_enhance" + ".jpg"
            enhance_image(input_image_path, output_image_path)
    

