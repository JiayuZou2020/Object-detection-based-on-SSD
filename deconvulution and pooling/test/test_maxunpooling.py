import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(img):
    npimg = img[0].data.numpy()
    npimg = ((npimg - npimg.min()) * 255 / (npimg.max() - npimg.min())).astype('uint8')
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
    plt.imsave("./maxunpooling_img.png", npimg)

if __name__ == '__main__':

    raw_img = cv2.imread("./data/car2.jpg")
    plt.imshow(raw_img)
    plt.show()
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)


    plt.imshow(raw_img)
    plt.show()
    plt.imsave("./original_img.png", raw_img)

    resized_img = cv2.resize(raw_img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor()
		#如果想要进行归一化则再添加下面这行代码
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_img = transform(resized_img).unsqueeze_(0)


    model = models.vgg16(pretrained=True).eval()
    for i, layer in enumerate(model.features):
        if isinstance(layer, torch.nn.MaxPool2d):
            layer.return_indices = True
	# 最大池化结果
    maxpooling_layer = model.features[4]
    maxpooling_result, indices = maxpooling_layer(input_img)
    visualize(maxpooling_result)

    kernel_size = maxpooling_layer.kernel_size
    stride = maxpooling_layer.stride
    padding = maxpooling_layer.padding
	# 反池化结果
    unpooling_layer = torch.nn.MaxUnpool2d(kernel_size, stride, padding)
    unpooling_result = unpooling_layer(maxpooling_result, indices)

    visualize(unpooling_result)
