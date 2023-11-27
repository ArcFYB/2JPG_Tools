'''
    根据一定的比例，将source_directory中的图片移动到floder1中,其余的移动到floder2中（图片.shuffle）
'''
import os
import sys
import shutil
import random

def split_images_by_percentage(source_folder, dest_folder_1, dest_folder_2, percentage, flag_move):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(dest_folder_1, exist_ok=True)
    os.makedirs(dest_folder_2, exist_ok=True)
    
    # 获取源文件夹中的所有图片文件
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 计算要移动的图片数量
    num_images = len(image_files)
    num_to_move_1 = int(num_images * (percentage / 100))
    num_to_move_2 = num_images - num_to_move_1

    # 打乱图片列表的顺序
    random.shuffle(image_files)

    # 将选择的图片移动到目标文件夹
    for i, image in enumerate(image_files):
        source_path = os.path.join(source_folder, image)
        if i < num_to_move_1:
            dest_path = os.path.join(dest_folder_1, image)
        else:
            dest_path = os.path.join(dest_folder_2, image)
        if flag_move:
            shutil.move(source_path, dest_path)
        else:
            shutil.copy(source_path, dest_path)
    print("已将" + str(num_to_move_1) + "张图片添加到" + dest_folder_1)
    print("已将" + str(num_to_move_2) + "张图片添加到" + dest_folder_2)
            

if __name__ == "__main__":
    # 使用示例
    source_directory = '/home/fiko/Code/YOLOP/yolop_dataset/label/lane_detection/4'
    # source_directory = sys.argv[1]
    # print(source_directory + "========================================")
    destination_directory_1 = '/home/fiko/Code/YOLOP/yolop_dataset/label/train/4'
    destination_directory_2 = '/home/fiko/Code/YOLOP/yolop_dataset/label/val/4'
    split_percentage = 70
    flag_move = False

    split_images_by_percentage(source_directory, destination_directory_1, destination_directory_2, split_percentage, flag_move)
