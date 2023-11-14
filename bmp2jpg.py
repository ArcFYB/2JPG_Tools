
# coding:utf-8
import os
from PIL import Image


path = 'C:\\Users\\Administrator\\Desktop\\ShipRSImageNet_V1\\VOC_Format\\JPEGImages'
path1 = 'C:\\Users\\Administrator\\Desktop\\ShipRSImage'
for file in os.listdir(path):
    if file.endswith('.bmp'):
        filename = os.path.join(path, file)
        new_name = path1 +'\\' + file[:-4] + '.jpg'
        img = Image.open(filename)
        image = img.convert('RGB')
        image.save(new_name)
        del img
        os.remove(filename)
        pass

