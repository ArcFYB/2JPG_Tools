import os
import shutil

def rename_sr_to_hr(source_folder):
    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件是否以 .sr.png 结尾
            if file.lower().endswith('sr.png'):
                # 构建原文件的完整路径和新文件名及路径
                original_file_path = os.path.join(root, file)
                new_file_name = file[:-6] + 'hr.png'  # 去掉'.sr'，加上'.hr'
                new_file_path = os.path.join(root, new_file_name)
                
                # 重命名文件（实质上是移动并改名，因为shutil.move可以实现改名效果）
                shutil.move(original_file_path, new_file_path)
                print(f"Renamed {file} to {new_file_name}")

# 使用方法：指定源文件夹路径
source_folder_path = '/home/fiko/Code/Super_Resolution/End2End_SR/experiments/NWPU_RESISC45/river/sr' 

rename_sr_to_hr(source_folder_path)