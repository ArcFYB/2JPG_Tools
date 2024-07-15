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
    stretched_image = np.clip((image_array - new_min) * (255.0 / (new_max - new_min)), 0, 255)
    return stretched_image.astype(np.uint8)

def process_block(block, max_value):
    block_uint8 = (block / max_value * 255).astype(np.uint8)
    return block_uint8

class ColorEnhancement:
    def __init__(self, dataset, value):
        self.dataset = dataset
        self.value = value
        self.rows = self.dataset.RasterYSize
        self.cols = self.dataset.RasterXSize
        self.bands = self.dataset.RasterCount
        if self.bands >= 3:
            self.image_setting()

    def image_setting(self):
        self.getband1 = self.dataset.GetRasterBand(1).ReadAsArray()
        self.getband2 = self.dataset.GetRasterBand(2).ReadAsArray()
        self.getband3 = self.dataset.GetRasterBand(3).ReadAsArray()
        self.getband4 = self.dataset.GetRasterBand(4).ReadAsArray()

    def NDVI(self):
        b4 = self.getband4.astype(np.float32)
        b3 = self.getband3.astype(np.float32)
        return (b4 - b3) / (b4 + b3)

    def vegetation_enhancement(self, ndvi):
        arr_ = self.getband2.copy()
        mask = ndvi > 0.2
        arr_[mask] = self.getband2[mask] * self.value + self.getband4[mask] * (1 - self.value)
        arr_[ndvi < 0.2] = self.getband2[ndvi < 0.2]
        return arr_

    def process(self):
        ndvi = self.NDVI()
        enhanced_bands = []
        for i in range(self.bands):
            if i == 1:
                data = self.vegetation_enhancement(ndvi)
                data_uint8 = linear_stretch(data, 2)
            else:
                data = self.dataset.GetRasterBand(i + 1).ReadAsArray()
                data_uint8 = linear_stretch(data, 2)
            enhanced_bands.append(data_uint8)
        return enhanced_bands

def linear_stretch(data, num=1):
    data_8bit = data
    data_8bit[data_8bit == -9999] = 0
    d2 = np.percentile(data_8bit, num)
    u98 = np.percentile(data_8bit, 100 - num)
    maxout = 255
    minout = 0
    data_8bit_new = minout + ((data_8bit - d2) / (u98 - d2)) * (maxout - minout)
    data_8bit_new[data_8bit_new < minout] = minout
    data_8bit_new[data_8bit_new > maxout] = maxout
    data_8bit_new = data_8bit_new.astype(np.uint8)
    return data_8bit_new

def crop_tiff_to_jpg(input_tiff_file, output_directory, block_size=256, enhancement_value=0.5):
    ds = gdal.Open(input_tiff_file)
    if ds is None:
        raise Exception("Unable to open TIFF file")

    width = ds.RasterXSize
    height = ds.RasterYSize
    os.makedirs(output_directory, exist_ok=True)
    max_value = 255
    total_blocks = (height // block_size) * (width // block_size)
    progress = tqdm(total=total_blocks, desc="Processing Image Blocks", unit="block")

    color_enhancer = ColorEnhancement(ds, enhancement_value)
    enhanced_bands = color_enhancer.process()

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_rgb = np.zeros((block_size, block_size, 3), dtype=np.uint8)
            complete = True
            for i, band_data in enumerate(enhanced_bands[:3]):  # Using only the first 3 bands (RGB)
                block = band_data[y:y + block_size, x:x + block_size]
                if block.shape[0] != block_size or block.shape[1] != block_size:
                    complete = False
                    break
                block_rgb[:, :, i] = process_block(block, max_value)
            if complete:
                block_rgb = stretch_dynamic_range(block_rgb)
                block_jpg_file = os.path.join(output_directory, f"block_{x}_{y}.jpg")
                cv2.imwrite(block_jpg_file, block_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            progress.update(1)

    progress.close()
    print("Conversion completed!!!")

if __name__ == "__main__":
    input_tiff_file = "/media/fiko/ARCFYB/GF2_area_test.tif"
    output_directory = "/media/fiko/ARCFYB/GF2"
    enhancement_value = 0.5
    crop_tiff_to_jpg(input_tiff_file, output_directory, enhancement_value=enhancement_value)
