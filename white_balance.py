import cv2
import numpy as np

def white_balance(image):
    # 将图像转换为Lab颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # 分离通道
    L, a, b = cv2.split(lab)
    
    # 计算a和b通道的均值
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    
    # 计算增益系数
    avg_a = 128 - avg_a
    avg_b = 128 - avg_b
    
    # 对a和b通道应用增益系数
    balanced_a = np.uint8(np.clip(a + avg_a, 0, 255))
    balanced_b = np.uint8(np.clip(b + avg_b, 0, 255))
    
    # 合并通道
    balanced_lab = cv2.merge([L, balanced_a, balanced_b])
    
    # 将图像转换回BGR颜色空间
    balanced_image = cv2.cvtColor(balanced_lab, cv2.COLOR_Lab2BGR)
    
    return balanced_image

# 读取原始图像
image = cv2.imread('/home/fiko/Downloads/20201114223523289.jpg')

# 进行白平衡处理
balanced_image = white_balance(image)

# 显示原始图像和处理后的图像
cv2.imshow('Original Image', image)
cv2.imshow('White Balanced Image', balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
