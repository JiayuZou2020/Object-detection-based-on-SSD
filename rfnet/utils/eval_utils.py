# -*- coding: utf-8 -*-
# @Time    : 2018-9-28 17:00
# @Author  : xylon
import cv2
import torch
import numpy as np
import time

import cv2
import torch
import numpy as np
import matplotlib as plt
from model.rf_des import HardNetNeiMask
from model.rf_det_so import RFDetSO
from model.rf_net_so import RFNetSO
from config import cfg
from utils.math_utils import distance_matrix_vector, pairwise_distances, ptCltoCr

def save_patchpair(patch_pair, name, save, size=None):
    bar = (
        range(patch_pair.size(0))
        if size is None
        else np.random.randint(patch_pair.size(0), size=size)
    )
    for i in bar:
        p1 = patch_pair[i][0].cpu().detach().numpy()  # (32, 32)
        p2 = patch_pair[i][1].cpu().detach().numpy()  # (32, 32)
        p_combi = np.concatenate((p1, p2), axis=1)
        cv2.imwrite(f"{save}/image/sppair_{name}_{i}.png", p_combi * 255)


def getAC(im1_ldes, im1_rdes):
    im1_distmat = distance_matrix_vector(im1_ldes, im1_rdes)
    row_minidx = im1_distmat.sort(dim=1)[1]  # (topk, topk)
    topk = row_minidx.size(0)
    s = row_minidx[:, :5]  # (topk, 5)
    flagim_index = s[:, 0].contiguous().view(-1) == torch.arange(topk).to(s.device)
    ac = flagim_index.float().mean()
    print('The first accuracy is',ac)
    return ac


def vis_descriptor_with_patches(endpoint, cfg, saveimg=False, imname=None):
    psize = cfg.PATCH.size
    save = cfg.TRAIN.SAVE

    def imarrange(imgs, topk):
        imgs = imgs.view(topk, -1, psize, psize)
        imgs = imgs.permute(0, 2, 1, 3).contiguous()
        imgs = imgs.view(topk * psize, -1)
        return imgs

    im1_ldes, im1_rdes = (
        endpoint["im1_ldes"],
        endpoint["im1_rdes"],
    )  # each is (topk, 128)
    im1_distmat = distance_matrix_vector(im1_ldes, im1_rdes)
    row_minidx = im1_distmat.sort(dim=1)[1]  # (topk, topk)
    topk = row_minidx.size(0)
    sorted = row_minidx[:, :5]  # (topk, 5)
    flagim_index = sorted[:, 0].contiguous().view(-1) == torch.arange(topk).to(
        sorted.device
    )
    ac = flagim_index.float().mean()
    if saveimg is True and imname is not None:
        # save image with batch op
        flagim_index = flagim_index.cpu().detach().numpy()
        im1_ppair = (endpoint["im1_ppair"] * cfg.IMAGE.STD + cfg.IMAGE.MEAN) * 255
        im1_lpatch, im1_rpatch = im1_ppair.chunk(
            chunks=2, dim=1
        )  # each is (topk, 1, 32, 32)
        tim = cv2.cvtColor(
            cv2.resize(cv2.imread("./tools/t.jpg"), (psize, psize)).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        fim = cv2.cvtColor(
            cv2.resize(cv2.imread("./tools/f.jpg"), (psize, psize)).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        flagim = (
            torch.from_numpy(np.stack((fim, tim), axis=0)).float().to(sorted.device)
        )
        flagr = imarrange(flagim[flagim_index], topk)  # (topk, 32, 32)
        anchor = imarrange(im1_lpatch.squeeze(), topk)  # (topk, 32, 32)
        target = imarrange(im1_rpatch.squeeze(), topk)  # (topk, 32, 32)
        sorted = sorted.contiguous().view(-1).cpu().detach().numpy()
        patches = imarrange(im1_rpatch[sorted].squeeze(), topk)  # (topk*5, 32, 32)
        im1_result = torch.cat((anchor, target, flagr, patches), dim=1)
        imname = imname + f"_ac{ac:05.2f}.png"
        cv2.imwrite(f"{save}/image/{imname}", im1_result.cpu().detach().numpy())
    print('The second accuracy is',ac)
    return ac


def eval_model(model, parsed_batch, DES_THRSH, COO_THRSH):
    """
    only support one bach size cause function ditance_matrix_vector
    """
    im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = (
        parsed_batch
    )

    # predict key points in each image and extract descripotor from each patch
    scale1, kp1, des1 = model.inference(im1_data, im1_info, im1_raw)
    scale2, kp2, des2 = model.inference(im2_data, im2_info, im2_raw)
    kp1w = ptCltoCr(kp1, homo12, scale2, right_imorint=None, clamp=False)[0]


    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.8)
    idx = predict_label.nonzero().view(-1)
    kp1 = kp1w
    mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
    mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2

    def to_cv2_kp(kp):
        # kp is like [batch_idx, y, x, channel]
        return cv2.KeyPoint(kp[2], kp[1], 0)

    def to_cv2_dmatch(m):
        return cv2.DMatch(m, m, m, m)

    #  DMatch.distance 
    #  DMatch.trainIdx 
    #  DMatch.queryIdx 
    #  DMatch.imgIdx 
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
    img1 = im1_data
    img2 = im2_data
    img1, img2 = reverse_img(img1), reverse_img(img2)
    keypoints1 = list(map(to_cv2_kp, mkp1))
    keypoints2 = list(map(to_cv2_kp, mkp2))
    DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))

    # matches1to2	Matches from the first image to the second one, which means that
    # keypoints1[i] has a corresponding point in keypoints2[matches[i]] .
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, DMatch, None)
    # cv2.imwrite("outImg.png", outImg)
    name = time.time()
    name = str(name)
    cv2.imwrite(name+'match.png',outImg)

    img_kps1 = np.copy(img1)
    img_kps1 = cv2.drawKeypoints(img1,keypoints1,img_kps1 ,flags = None)
    cv2.imwrite("img_kps3.png", img_kps1)
    img_kps2 = np.copy(img2)
    img_kps2 = cv2.drawKeypoints(img2,keypoints2,img_kps2 ,flags = None)
    cv2.imwrite("img_kps4.png", img_kps2)

    _, _, maxh, maxw = im2_data.size()
    visible = kp1w[:, 2].lt(maxw) * kp1w[:, 1].lt(maxh)

    TPNN, PNN = nearest_neighbor_match_score(des1, des2, kp1w, kp2, visible, COO_THRSH)
    TPNNT, PNNT = nearest_neighbor_threshold_match_score(
        des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH
    )
    TPNNDR, PNNDR = nearest_neighbor_distance_ratio_match_score(
        des1, des2, kp1w, kp2, visible, COO_THRSH
    )
    print('TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR are as followings:')
    print(TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR)
    return TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR


def nearest_neighbor_match_score(des1, des2, kp1w, kp2, visible, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)

    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correct_match_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(visible.sum().item(), 1)

    return correct_matches, predict_matches


def nearest_neighbor_threshold_match_score(
    des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH
):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)
    predict_label = nn_value.lt(DES_THRSH) * visible

    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)

    return correct_matches, predict_matches


def threshold_match_score(des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    visible = visible.unsqueeze(-1).repeat(1, des_dist_matrix.size(1))
    predict_label = des_dist_matrix.lt(DES_THRSH) * visible

    coo_dist_matrix = pairwise_distances(kp1w[:, 1:3].float(), kp2[:, 1:3].float())
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)
    correspond_matches = max(correspondences_label.sum().item(), 1)

    return correct_matches, predict_matches, correspond_matches


def nearest_neighbor_distance_ratio_match(des1, des2, kp2, threshold):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    sorted, indices = des_dist_matrix.sort(dim=-1)
    Da, Db, Ia = sorted[:, 0], sorted[:, 1], indices[:, 0]
    DistRatio = Da / Db
    predict_label = DistRatio.lt(threshold)
    nn_kp2 = kp2.index_select(dim=0, index=Ia.view(-1))
    return predict_label, nn_kp2


def nearest_neighbor_distance_ratio_match_score(
    des1, des2, kp1w, kp2, visible, COO_THRSH, threshold=0.7
):
    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(
        des1, des2, kp2, threshold
    )

    predict_label = predict_label * visible

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)

    return correct_matches, predict_matches
