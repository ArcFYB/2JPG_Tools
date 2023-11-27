'''
    根据json绘制黑白二值图
'''
import os
from PIL import Image, ImageDraw
import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_image(data, output_folder, json_file_path):
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    image = Image.new('L', (image_width, image_height), 0)  # 创建黑色图像
    draw = ImageDraw.Draw(image)

    for shape in data['shapes']:
        if shape['shape_type'] == 'linestrip':
            points = [(int(x), int(y)) for x, y in shape['points']]
            draw.line(points, fill=255, width=16)

    # 获取输入 JSON 文件的文件名（不带后缀）
    file_name = os.path.splitext(os.path.basename(json_file_path))[0]

    # 指定输出图像文件夹
    output_path = os.path.join(output_folder, f'{file_name}_output_image.jpg')
    
    image.save(output_path)

def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_file_path = os.path.join(input_folder, filename)
            json_data = load_json(json_file_path)
            create_image(json_data, output_folder, json_file_path)
            
print("completed!!!")

if __name__ == "__main__":
    input_folder_path = "/home/fiko/Code/YOLOP/yolop_dataset/label/乡村车道线标注数据原件/01"
    output_folder_path = "/home/fiko/Code/YOLOP/yolop_dataset/label/lane_detection/1"
    
    process_folder(input_folder_path, output_folder_path)

