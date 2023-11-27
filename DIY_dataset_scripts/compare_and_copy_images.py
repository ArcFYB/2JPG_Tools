'''
    比较floder1与floder2中图片名称前四位是否相同
    if true：
        copyfile from folder2_path to destination_path
'''
import os
import shutil

def compare_and_copy_images(folder1, folder2, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(destination_folder, exist_ok=True)

    # 获取文件夹中的文件名列表
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # 获取文件夹1中文件名的前四位
    filenames1 = {filename[:4] for filename in files1}

    # 将符合条件的文件复制到目标文件夹
    for filename2 in files2:
        if filename2[:4] in filenames1:
            source_path = os.path.join(folder2, filename2)
            destination_path = os.path.join(destination_folder, filename2)
            shutil.copyfile(source_path, destination_path)
    print("completed!!!")

# 使用示例

folder1_path = '/home/fiko/Code/YOLOP/DATASET/images/val' # RGB图像
folder2_path = '/home/fiko/Code/YOLOP/yolop_dataset/label/lane_detection/4' # 黑白二值图
destination_path = '/home/fiko/Code/YOLOP/DATASET/ll_seg_annotations/val'
'''
folder1_path = '/home/fiko/Code/YOLOP/DATASET/images/train' # RGB图像
folder2_path = '/home/fiko/Code/YOLOP/yolop_dataset/label/lane_detection/4' # 黑白二值图
destination_path = '/home/fiko/Code/YOLOP/DATASET/ll_seg_annotations/train'
'''


compare_and_copy_images(folder1_path, folder2_path, destination_path)
