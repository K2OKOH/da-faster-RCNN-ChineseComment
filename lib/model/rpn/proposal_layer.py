from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()
        '''
        # scales        ->  [4,8,16,32]
        # ratios        ->  [0.5,1,2]
        # feat_stride   ->  16,vgg下采样倍数
        '''
        self._feat_stride = feat_stride
        # 产生anchor并转化为张量,anchor是二维 ndarray,每行为一个框()
        # 但这里只在角落产生了一组(12个)anchor
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
            ratios=np.array(ratios))).float()
        # 记录anchor数量
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):
        '''
        输入:
            是个tuple(),元素如下
            rpn_cls_prob    ->  torch.size([1, 24, H, W]), 每层的像素是fg/bg的概率
            rpn_bbox_pred   ->  预测bbox的feature_map(大小:[1, 48, H, W])
            im_info         ->  图片的大小H, W, ratio
            cfg_key         ->  'TRAIN' or 'TEST'
        输出:
            output          ->  size([1, num_proposal, 5])
        '''

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        # 读入数据
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N  # 12000  NMS前保留的proposal
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # 2000   NMS后保留的proposal
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH     # 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE       # 8

        # 这里就是1张图
        batch_size = bbox_deltas.size(0)

        # feature_map的尺寸
        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        '''shift_x !! 例子
        [[   0   16   32 ... 1152 1168 1184]
         [   0   16   32 ... 1152 1168 1184]
         [   0   16   32 ... 1152 1168 1184]
         ...
         [   0   16   32 ... 1152 1168 1184]
         [   0   16   32 ... 1152 1168 1184]
         [   0   16   32 ... 1152 1168 1184]]
        '''
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        '''np.vstack(...) -> 然后进行转置
        [[   0   16   32 ... 1152 1168 1184]
         [   0    0    0 ...  576  576  576]
         [   0   16   32 ... 1152 1168 1184]
         [   0    0    0 ...  576  576  576]]
        '''
        # type_as(scores)按照scores的类型来定义
        shifts = shifts.contiguous().type_as(scores).float()

        # 一个位置anchor的数量
        A = self._num_anchors
        # K = W*H -> 2775 = 75*37
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        # 改变_anchor矩阵的形状(4,12) -> (1,12,4)
        # 维度不相同时，对维度为1的进行广播操作 (1,12,4) + (W*H,1,4) (W*H,12,4) -> 对anchor进行平铺
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # 改变_anchor矩阵的形状(K,12,4) -> (batch_size,K*12,4) 把第一维最为batch中图片的序号(本程序中就是1)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        # bbox_deltas = feature_map(大小:[1, 48, H, W])
        # 进行维度转换 -> (大小:[1, 48, H, W] -> [1, H, W, 48]) (48的含义是 -> 一个anchor12个bbox每个4个坐标)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        # 改变size [1, H, W, 48] -> [1, H*W*12, 4]  (一共H*W*12个bbox)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        # scores(size:[1,24,H,W] -> [1,H,W,24] -> [1,H*W*24])
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        '''
        进行每个anchor(FeatureMap上每个像素12个)
        输入:
            anchors     :候选框       ->  size:(1, H*W*12, 4)
            bbox_deltas :FeatureMap  ->  size:(1, H*W*12, 4)
            batch_size  :fixed       ->  1
        输出:
            proposals   :经过预测的anchor -> size:(1, H*W*12, 4)
        '''
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        # 把预测之后的Anchor的坐标限制在,图像的大小之内
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)
        
        # _, order = torch.sort(scores_keep, 1, True)
        
        scores_keep = scores
        proposals_keep = proposals
        # 对scores_keep(FeatureMap是前后景的概率)排序(对第一维度,从大到小,对第1维)
        # scores_keep -> size(1, H*W*12)
        _, order = torch.sort(scores_keep, 1, True)

        # 创建和scores类型相同的size(1, 2000, 5)的张量,初始化为0
        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        # 这里batch_size恒为1
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            # proposals_keep是经过预测的顶点坐标 -> size:(1, H*W*12, 4)
            # 取出第一维度    proposals_single -> 大小(H*W*12, 4)       scores_single -> 大小(H*W*12)
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            # order_single -> 大小size(H*W*12)
            order_single = order[i]

            # pre_nms_topN = 12000, 取前12000个最可能是fg的anchor
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            '''
                proposals_single    ->  size:(12000, 4)
                scores_single       ->  size:(12000, 1)
                cat后    -> size:(12000, 5)
                nms_thresh = 0.7
                nms()   在nms文件夹中,gpu操作复杂可以看cpu的原理
                返回: 索引号
            '''
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                # 找到前post_nms_topN个fg置信度的索引号
                keep_idx_i = keep_idx_i[:post_nms_topN]
            # 保留前post_nms_topN个fg置信度的anchor
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            # 记录剩余的anchor(proposal)的数量
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            # 写入输出张量
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
