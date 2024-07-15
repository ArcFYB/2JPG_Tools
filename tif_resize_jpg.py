from osgeo import gdal
import os
import cv2
import numpy as np
from tqdm import tqdm

def crop_tiff_to_jpg(input_tiff_file, output_directory, block_size=256):
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

    # 遍历TIFF图像并切割成128x128的块
    pbar = tqdm(total=(width // block_size) * (height // block_size))
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 提取图像块
            block_rgb = rgb_image[y:y+block_size, x:x+block_size, :]

            # 如果图像块中存在像素值为0，则跳过保存
            if np.any(block_rgb == 0):
                continue

            # 保存图像块为JPEG格式
            block_jpg_file = os.path.join(output_directory, f"block_{x}_{y}.jpg")
            cv2.imwrite(block_jpg_file, block_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    input_tiff_file = '/media/fiko/ARCFYB/GF2_area_test.tif'  # 输入TIFF文件的路径
    output_directory = '/media/fiko/ARCFYB/GF2'  # 保存图像块的目录

    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)

    crop_tiff_to_jpg(input_tiff_file, output_directory)
