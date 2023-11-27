'''
    重命名floder下的图片为，原名字的前四位.jpg
'''
import os

def rename_images_with_first_four_characters(folder_path):
    # 获取文件夹中的所有图片文件
    all_files = os.listdir(folder_path)
    for file_name in all_files:
        if not file_name.endswith('.jpg'):
            old_name = os.path.join(folder_path, file_name)
            new_name = os.path.join(folder_path, f"{file_name}.jpg")  # 添加'.jpg'作为文件的后缀名
            os.rename(old_name, new_name)
    
    # ------------------------------------------------------------------------------------------
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    # 重命名图片文件
    for image in image_files:
        if image.endswith('.jpg'):
            old_name = os.path.join(folder_path, image)
            new_name = os.path.join(folder_path, f"{image[:4]}.jpg")  # 使用前四位作为新文件名，并添加'.jpg'
            os.rename(old_name, new_name)
        elif image.endswith('.png'):
            old_name = os.path.join(folder_path, image)
            new_name = os.path.join(folder_path, f"{image[:4]}.png")  # 使用前四位作为新文件名，并添加'.png'
            os.rename(old_name, new_name)
        
    print("completed!!!")


# 使用示例
folder_path = '/home/fiko/Code/YOLOP/DATASET/ll_seg_annotations/train'

rename_images_with_first_four_characters(folder_path)
