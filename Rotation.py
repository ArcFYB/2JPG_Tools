'''
去除jpg格式遥感影像黑边，并旋转
'''

import cv2
import numpy as np

# 1. 读取图像
img = cv2.imread('/home/fiko/Code/Super_Resolution/Image-Super-Resolution-via-Iterative-Refinement/dataset/tif_dataset/airport_MUX/airport.jpg')

# 2. 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 找到最大的轮廓
# 使用Canny边缘检测算法检测图像中的边缘
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 使用霍夫变换检测图像中的直线
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# 获取检测到的直线中最长的一条
longest_line = None
max_length = 0

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        length = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        if length > max_length:
            max_length = length
            longest_line = line

# 获取直线的端点坐标
for rho, theta in longest_line:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

# 计算直线的角度
angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

# 4. 获取最小外接矩形
h, w = img.shape[:2]
rect = cv2.minAreaRect(np.array([(x, y) for x in range(w) for y in range(h) if edges[y, x] > 0]))
box = cv2.boxPoints(rect)
box = np.int0(box)

# 5. 旋转原始图像以校正矩形
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (width, height))

# 6. 显示中间过程（用于可解释性研究）
# cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
# cv2.imshow('edge', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 7. 裁剪并保存结果图像
output = warped[0:height, 0:width]
cv2.imwrite('./output.jpg', output)
