import os
import shutil

def copy_images(source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件是否为图片（这里以常见的图片格式为例，你可以根据需要添加更多格式）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构建源文件的完整路径和目标文件的路径
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(destination_folder, file)
                
                # 复制文件到目标文件夹
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {file} to {destination_folder}")

# 使用方法：指定源文件夹路径和目标文件夹路径
source_folder_path = '/home/fiko/Code/KAIR/NWPU_RESISC45/river_32_128/msrresnet_psnr/images'
destination_folder_path = '/home/fiko/Code/KAIR/NWPU_RESISC45/river_32_128/msrresnet_psnr/SR_images'

copy_images(source_folder_path, destination_folder_path)