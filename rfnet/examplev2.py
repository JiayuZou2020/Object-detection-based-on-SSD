# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:20
# @Author  : xylon
import cv2
import torch
import random
import argparse
import numpy as np
import matplotlib as plt

import time
import datetime

from utils.common_utils import gct
from utils.eval_utils import nearest_neighbor_distance_ratio_match
from model.rf_des import HardNetNeiMask
from model.rf_det_so import RFDetSO
from model.rf_net_so import RFNetSO
from config import cfg

import warnings
warnings.filterwarnings("ignore") 

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="example")
    # parser.add_argument("--imgpath", default=None, type=str)  # image path
    # parser.add_argument("--resume", default=None, type=str)  # model path
    # args = parser.parse_args()
    imgpath = 'E:/Document/senior_four/paper_code/rfnet/material'
    resume = 'E:/Document/senior_four/paper_code/runs/01_02_20_31/model/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar'
# 输入函数参数
    print(f"{gct()} : start time")

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    print(f"{gct()} : model init")
    det = RFDetSO(
        cfg.TRAIN.score_com_strength,
        cfg.TRAIN.scale_com_strength,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        cfg.TRAIN.TOPK,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
        cfg.MODEL.KSIZE,
        cfg.MODEL.padding,
        cfg.MODEL.dilation,
        cfg.MODEL.scale_list,
    )
    des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
    model = RFNetSO(
        det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
    )

    print(f"{gct()} : to device")
    device = torch.device("cuda")
    model = model.to(device)
    # resume = args.resume
    print(f"{gct()} : in {resume}")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])

    ###############################################################################
    # detect and compute
    ###############################################################################
    # img1_path, img2_path = args.imgpath.split("@")
    img1_path = 'E:/Document/senior_four/paper_code/rfnet/material/img2.png'
    img2_path = 'E:/Document/senior_four/paper_code/rfnet/material/img3.png'
    kp1, des1, img1 = model.detectAndCompute(img1_path, device, (240, 320))
    kp2, des2, img2 = model.detectAndCompute(img2_path, device, (240, 320))

    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 1.5)
    idx = predict_label.nonzero().view(-1)
    mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
    mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2
    # 其中mkp1和mkp2分别是image1和image2当中预测的关键点，可以打印出来做可视化效果
    def to_cv2_kp(kp):
        # kp is like [batch_idx, y, x, channel]
        return cv2.KeyPoint(kp[2], kp[1], 0)

    def to_cv2_dmatch(m):
        return cv2.DMatch(m, m, m, m)
    # DMatch函数返回结果的四个参数意义如下：
    # • DMatch.distance - 描述符之间的距离。越小越好。
    # • DMatch.trainIdx - 目标图像中描述符的索引。
    # • DMatch.queryIdx - 查询图像中描述符的索引。
    # • DMatch.imgIdx - 目标图像的索引。
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

    img1, img2 = reverse_img(img1), reverse_img(img2)
    keypoints1 = list(map(to_cv2_kp, mkp1))
    keypoints2 = list(map(to_cv2_kp, mkp2))
    DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))
    
    # cv2.drawMatches()表示对关键点进行连线操作
    # matches1to2	Matches from the first image to the second one, which means that
    # keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, DMatch, None)
    # cv2.imwrite("outImg_1.png", outImg)

    name = time.time()
    name = str(name)
    cv2.imwrite(name+'match.png',outImg)

    img_kps1 = np.copy(img1)
    img_kps1 = cv2.drawKeypoints(img1,keypoints1,img_kps1 ,flags = None)
    cv2.imwrite("img_kps1.png", img_kps1)
    img_kps2 = np.copy(img2)
    img_kps2 = cv2.drawKeypoints(img2,keypoints2,img_kps2 ,flags = None)
    cv2.imwrite("img_kps2.png", img_kps2)
    # 将特征点显示出来
    # cv2.imshow('Keypoints',img_kps1 )
    # cv2.waitKey()
