import os
from PIL import Image

dirname_read=r"C:\Users\Administrator\Desktop\Deep-Learning-master\Pytorch-Seg\lesson-2\data\train\label/"  # 注意后面的斜杠
dirname_write="F:/Pytorch-UNet/data/masks/"
names=os.listdir(dirname_read)
count=0
for name in names:
    img=Image.open(dirname_read+name)
    name=name.split(".")
    if name[-1] == "png":
        name[-1] = "jpg"
        name = str.join(".", name)
        #r,g,b,a=img.split()              
        #img=Image.merge("RGB",(r,g,b))   
        to_save_path = dirname_write + name
        img = img.convert('RGB')#RGBA意思是红色，绿色，蓝色，Alpha的色彩空间，Alpha指透明度。而JPG不支持透明度，所以要么丢弃Alpha,要么保存为.png文件
        img.save(to_save_path)
        count+=1
        print(to_save_path, "------conut：",count)
    else:
        continue
