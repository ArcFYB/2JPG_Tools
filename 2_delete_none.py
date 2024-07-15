from PIL import Image
import os

def black_pixel_ratio(image_path):
    """计算图片中黑色像素的比例"""
    with Image.open(image_path) as img:
        img = img.convert("L")
        data = img.getdata()
        total_pixels = len(data)
        black_pixels = sum(1 for pixel in data if pixel == 0)
        return black_pixels / total_pixels

def delete_black_images(directory, threshold=0.1):
    """遍历目录，删除黑色像素比例超过threshold的图片"""
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            ratio = black_pixel_ratio(file_path)
            if ratio > threshold:
                print(f"删除黑色像素比例超过{threshold*100}%的图片: {file_path}")
                os.remove(file_path)


# 使用示例
directory_to_check = '/media/fiko/ARCFYB/GF2'  # 设置你要检查的目录路径
delete_black_images(directory_to_check)