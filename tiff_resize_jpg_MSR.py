'''
1、将tiff裁剪为bolck_size大小的图片，保存为jpg
2、颜色处理： 进行MSR算法
3、保存未处理jpg和色彩增强后的jpg
'''

import numpy as np
import cv2
from osgeo import gdal
import os

def replace_zeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_r = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replace_zeroes(img)
        l_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        l_blur = replace_zeroes(l_blur)
        dst_img = cv2.log(img/255.0)
        dst_l_blur = cv2.log(l_blur/255.0)
        dst_ixl = cv2.multiply(dst_img, dst_l_blur)
        log_r += weight * cv2.subtract(dst_img, dst_ixl)

    dst_r = cv2.normalize(log_r, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_r)
    return log_uint8

def crop_tiff_to_jpg(input_tiff_file, output_directory, block_size=128):
    # 打开TIFF文件
    ds = gdal.Open(input_tiff_file)

    if ds is None:
        raise Exception("无法打开TIFF文件")

    # 获取TIFF图像的宽度和高度
    width = ds.RasterXSize
    height = ds.RasterYSize

    # 获取前三个波段的数据
    blue_band = ds.GetRasterBand(1).ReadAsArray()
    green_band = ds.GetRasterBand(2).ReadAsArray()
    red_band = ds.GetRasterBand(3).ReadAsArray()

    # 转换数据类型为uint8
    max_value = np.max([blue_band, green_band, red_band], where=(blue_band != 0), initial=1)
    blue_band_uint8 = (blue_band / max_value * 255).astype(np.uint8)
    green_band_uint8 = (green_band / max_value * 255).astype(np.uint8)
    red_band_uint8 = (red_band / max_value * 255).astype(np.uint8)

    # 创建RGB图像
    rgb_image = np.stack([blue_band_uint8, green_band_uint8, red_band_uint8], axis=-1)
    cv2.imwrite('/home/fiko/Code/Super_Resolution/Image-Super-Resolution-via-Iterative-Refinement/dataset/tif_dataset/airport_MUX/result.jpg', rgb_image)
    
    # 使用MSR算法进行颜色平衡
    scales = [15, 101, 301]
    b_gray = MSR(blue_band_uint8, scales)
    g_gray = MSR(green_band_uint8, scales)
    r_gray = MSR(red_band_uint8, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    cv2.imwrite('/home/fiko/Code/Super_Resolution/Image-Super-Resolution-via-Iterative-Refinement/dataset/tif_dataset/airport_MUX/result_enhanced.jpg', result)

    # 遍历TIFF图像并切割成128x128的块
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 提取图像块
            block_rgb = result[y:y+block_size, x:x+block_size, :]

            # 如果图像块中存在像素值为0，则跳过保存
            if np.any(block_rgb == 0):
                continue

            # 保存图像块为JPEG格式
            block_jpg_file = os.path.join(output_directory, f"block_{x}_{y}.jpg")
            cv2.imwrite(block_jpg_file, block_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print("save "+ block_jpg_file + " to " + output_directory)
            
    print("Convert image completed!!!")

if __name__ == "__main__":
    input_tiff_file = "/home/fiko/Code/Super_Resolution/Image-Super-Resolution-via-Iterative-Refinement/dataset/tif_dataset/airport_MUX/airport.tif"  # 输入TIFF文件的路径
    output_directory = "/home/fiko/Code/Super_Resolution/Image-Super-Resolution-via-Iterative-Refinement/dataset/tif_dataset/airport_retinex_jpg"  # 保存图像块的目录

    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)

    crop_tiff_to_jpg(input_tiff_file, output_directory)
