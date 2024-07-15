import cv2
import numpy as np
import os
import shutil
from random import sample

def process_images(source_dir, output_dir_train, output_dir_val, val_ratio=0.2):
    # 创建输出目录
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_val, exist_ok=True)

    # 获取源目录中所有jpg文件
    images = [file for file in os.listdir(source_dir) if file.endswith('.jpg')]
    valid_images = []
            
    # 遍历所有图像并删除包含黑色像素的图像
    for image_name in images:
        image_path = os.path.join(source_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # 计算图像中黑色像素的数量
        black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))

        # 计算图像的总像素数量
        total_pixels = img.shape[0] * img.shape[1]

        # 计算黑色像素占总像素的比例
        black_pixel_ratio = black_pixels / total_pixels

        # 如果黑色像素比例超过10%，删除图像
        if black_pixel_ratio > 0.10:
            os.remove(image_path)
        else:
            valid_images.append(image_name) 

    # 从有效的图像中随机选择一部分作为验证集
    num_val = int(len(valid_images) * val_ratio)
    val_images = sample(valid_images, num_val)
    train_images = list(set(valid_images) - set(val_images))

    # 将训练集和验证集图像移动到相应的文件夹
    for image_name in train_images:
        source_path = os.path.join(source_dir, image_name)
        destination_path = os.path.join(output_dir_train, image_name)
        shutil.move(source_path, destination_path)
    
    for image_name in val_images:
        source_path = os.path.join(source_dir, image_name)
        destination_path = os.path.join(output_dir_val, image_name)
        shutil.move(source_path, destination_path)

    print(f"Processed {len(train_images)} training images and {len(val_images)} validation images.")

if __name__ == "__main__":
    source_directory = '/home/fiko/Code/DATASET/508_Dataset/TongXiang' 
    train_directory = '/home/fiko/Code/DATASET/508_Dataset/TX_Train_512'
    validation_directory = '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512'
    process_images(source_directory, train_directory, validation_directory)
