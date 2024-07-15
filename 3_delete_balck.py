import cv2
import os
import numpy as np

def delete_images_with_black_pixels(directory):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):  # 确保处理的是 JPG 文件
            file_path = os.path.join(directory, filename)
            img = cv2.imread(file_path)
            if img is None:
                continue
            # 检查图像是否包含黑色像素
            if np.any(img == 0):
                os.remove(file_path)  # 删除包含黑色像素的图像
                print(f"Deleted {filename} because it contains black pixels.")

if __name__ == "__main__":
    directory = '/home/fiko/Code/DATASET/508_Dataset/TongXiang'  # 设置你的文件夹路径
    delete_images_with_black_pixels(directory)
