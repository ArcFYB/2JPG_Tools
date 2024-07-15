from osgeo import gdal
import os
import cv2
import numpy as np
from tqdm import tqdm

def stretch_dynamic_range(image, percent=0.001):
    image_array = np.array(image)
    flattened_image = image_array.flatten()
    sorted_pixels = np.sort(flattened_image)
    min_idx = int(len(sorted_pixels) * percent)
    max_idx = int(len(sorted_pixels) * (1 - percent))
    new_min = sorted_pixels[min_idx]
    new_max = sorted_pixels[max_idx]
    stretched_image = np.where(image_array < new_min, 0,
                               np.where(image_array > new_max, 255,
                                        (image_array - new_min) * (255.0 / (new_max - new_min))))
    stretched_image = stretched_image.astype(np.uint8)
    return stretched_image

def crop_tiff_to_jpg(input_tiff_file, output_directory, block_size=256):
    ds = gdal.Open(input_tiff_file)
    if ds is None:
        raise Exception("无法打开TIFF文件")

    width = ds.RasterXSize
    height = ds.RasterYSize
    blue_band = ds.GetRasterBand(3).ReadAsArray()
    green_band = ds.GetRasterBand(2).ReadAsArray()
    red_band = ds.GetRasterBand(1).ReadAsArray()

    max_value = np.max([blue_band, green_band, red_band], where=(blue_band != 0), initial=1)
    blue_band_uint8 = (blue_band / max_value * 255).astype(np.uint8)
    green_band_uint8 = (green_band / max_value * 255).astype(np.uint8)
    red_band_uint8 = (red_band / max_value * 255).astype(np.uint8)

    rgb_image = np.stack([blue_band_uint8, green_band_uint8, red_band_uint8], axis=-1)
    rgb_image = stretch_dynamic_range(rgb_image)
    # cv2.imwrite('/home/fiko/Code/DATASET/TX_0.5/TX_Autolevel.jpg', rgb_image)

    total_blocks = ((width // block_size) + 1) * ((height // block_size) + 1)
    pbar = tqdm(total=total_blocks, desc="Processing blocks")

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_rgb = rgb_image[y:y+block_size, x:x+block_size, :]
            if np.any(block_rgb == 0):
                pbar.update(1)
                continue

            block_jpg_file = os.path.join(output_directory, f"block_{x}_{y}.jpg")
            cv2.imwrite(block_jpg_file, block_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            print("Saved " + block_jpg_file + " to " + output_directory)
            pbar.update(1)

    pbar.close()
    print("Convert image completed!!!")

if __name__ == "__main__":
    input_tiff_file = "/media/fiko/ARCFYB/GF2_area_test.tif"
    output_directory = "/media/fiko/ARCFYB/GF2"
    os.makedirs(output_directory, exist_ok=True)
    crop_tiff_to_jpg(input_tiff_file, output_directory)
