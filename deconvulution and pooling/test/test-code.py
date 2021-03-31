import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取原始图片
raw_img = cv2.imread("./data/car2.jpg")
# 将图片变成固定的尺寸224*224
resized_img = cv2.resize(raw_img, (224, 224))
# 分别将原始图片、色彩通道变换图片和修改尺寸后的图片画出来
plt.title('Oringinal image')
plt.imshow(raw_img)
plt.show()

plt.title('Color-changed image')
cg_color_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
plt.imshow(cg_color_img)
plt.show()

plt.title('Resized image')
plt.imshow(resized_img)
plt.show()
plt.imsave("./resized_img.png", resized_img)