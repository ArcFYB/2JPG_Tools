import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import match_histograms

# 读取两张图片
image1 = io.imread('/home/fiko/Code/Super_Resolution/End2End_SR/experiments/NWPU_240627_160650/results/2000/70000_1_hr.png')
image2 = io.imread('/home/fiko/Code/Super_Resolution/End2End_SR/experiments/NWPU_240627_160650/results/2000/70000_1_sr.png' )


# 进行直方图匹配
matched_image = match_histograms(image2, image1, channel_axis=-1)

# 显示结果
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].imshow(image1)
axes[0].set_title('Reference Image')
axes[0].axis('off')

axes[1].imshow(image2)
axes[1].set_title('Original Image')
axes[1].axis('off')

axes[2].imshow(matched_image)
axes[2].set_title('Histogram Matched Image')
axes[2].axis('off')

plt.show()

