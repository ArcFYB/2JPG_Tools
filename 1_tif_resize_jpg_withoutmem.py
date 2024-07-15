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

def crop_tiff_to_jpg(input_tiff_file, output_directory, block_size=256):
    ds = gdal.Open(input_tiff_file)
    if ds is None:
        raise Exception("Unable to open TIFF file")

    width = ds.RasterXSize
    height = ds.RasterYSize

    os.makedirs(output_directory, exist_ok=True)

    # Calculate the scale for uint8 conversion
    max_value = 255  # You might need to adjust this based on your data range

    total_blocks = (height // block_size) * (width // block_size)
    progress = tqdm(total=total_blocks, desc="Processing Image Blocks", unit="block")

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_rgb = np.zeros((block_size, block_size, 3), dtype=np.uint8)
            complete = True

            for i, band_index in enumerate([3, 2, 1]):  # Assuming RGB order in file
                band = ds.GetRasterBand(band_index)
                block = band.ReadAsArray(x, y, block_size, block_size)
                if block is None or block.shape[0] != block_size or block.shape[1] != block_size:
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
    crop_tiff_to_jpg(input_tiff_file, output_directory)
