import os
from PIL import Image

def process_image(img, roi):
    x, y, width, height = roi
    # 裁剪感兴趣区域
    img_cropped = img.crop((x, y, x + width, y + height))
    return img_cropped

def process_images_in_folder(input_folder, output_folder, roi):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            img_cropped = process_image(img, roi)
            
            # 保存处理后的图像
            cropped_path = os.path.join(output_folder, f"cropped_{filename}")
            img_cropped.save(cropped_path)
            print(f"Processed and saved: {filename}")

# 使用示例
input_folder = '/home/fiko/Code/DATASET/508_Dataset/TX_Valdation_512'  # 输入文件夹路径
output_folder = '/home/fiko/Code/DATASET/508_Dataset/TX_compare_128'  # 输出文件夹路径
roi = (128, 256, 128, 128)  # 感兴趣区域的坐标和大小 (x, y, width, height)

process_images_in_folder(input_folder, output_folder, roi)
