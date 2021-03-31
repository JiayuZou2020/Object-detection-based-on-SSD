import cv2
import torch
import random
import argparse
import numpy as np
import matplotlib as plt


def reverse_img(img):
    """
    reverse image from tensor to cv2 format
    :param img: tensor
    :return: RBG image
    """
    img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)  # change to opencv format
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
    return img

img1 = cv2.imread('E:/Document/senior_four/paper_code/rfnet/material/img2.png')
cv2.imshow('Keypoints1',img1 )
cv2.waitKey()

img2 = reverse_img(img1)
cv2.imshow('Keypoints2',img2 )
cv2.waitKey()

