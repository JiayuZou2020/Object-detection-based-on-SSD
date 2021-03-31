import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(img):
    # 将图片转成numpy数组
    npimg = img[0].data.numpy()
	# 将数组归一化
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype('uint8')
	# 调换图片色彩通道
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
    plt.imsave("./deconvolution_img.png", npimg)

if __name__ == '__main__':
	# 图片读取、通道变换、resize
    raw_img = cv2.imread("./data/car2.jpg")
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(raw_img, (224, 224))
	# transform用来做归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(resized_img).unsqueeze_(0)
	# 加载VGG网络
    model = models.vgg16(pretrained=True).eval()
    print('Layers and their parameters in VGG16 are as follows:\n',model.features)
    conv_layer = model.features[0]
    conv_result = conv_layer(input_img)

    in_channels = conv_layer.out_channels
    out_channels = input_img.shape[1]
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    print('Parameters of in_channels,out_channels,kernel_size,stride,padding are as following:\n',in_channels,out_channels,kernel_size,stride,padding)
    # 反卷积运算
	deconv_layer = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    deconv_layer.weight = conv_layer.weight
   
    deconv_result = deconv_layer(conv_result)

    visualize(deconv_result)
