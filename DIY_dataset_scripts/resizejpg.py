'''
    将图片resize为固定大小
'''
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=1280,height=720):
    img=Image.open(jpgfile)   
    new_img=img.resize((width,height),Image.BILINEAR)   
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
for jpgfile in glob.glob("/home/fiko/Code/YOLOP/ORG_Dataset/ll_seg_annotations/val/*.jpg"): #返回12文件夹下所有的jpg路径
    convertjpg(jpgfile,"/home/fiko/Code/YOLOP/DATASET/ll_seg_annotations/val") #返回的是111文件夹下下个文件的所有路径