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
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()
        '''
        # scales        ->  [4,8,16,32]
        # ratios        ->  [0.5,1,2]
        # feat_stride   ->  16,vgg下采样倍数
        '''
        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        # 不允许超过边界
        self._allowed_border = 0  # default is 0

    '''
    输入:
        rpn_cls_score   :FeatureMap经过一层分类的卷积后  ->  [1, 24, H, W]
        gt_boxes   ->  [1, 数量, 5]
        im_info    ->  [1,3]
        num_boxes  ->  [1]
    '''
    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        # 依然是1
        batch_size = gt_boxes.size(0)

        #featuremap的H和W, 和proposal_layer类似
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        # _feat_stride   ->  16, vgg下采样倍数
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # all_anchors -> size:(W*H*12,4)
        all_anchors = all_anchors.view(K * A, 4)

        # 总共anchor的个数
        total_anchors = int(K * A)
        # keep 大小([33300 = W*H*12]) -> 判断每一个anchor是否超出边界
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        # 找出 不是0(不超越边界)的索引
        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        # 保留不是0的部分
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        # labels 大小(batch_size=1, anchor数量inds_inside),初始化为-1
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        # bbox_inside_weights 大小(batch_size=1, inds_inside),初始化为0
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        # gt_boxes   ->  [1, 数量, 5]
        # anchors   ->  [anchor数量, 4]
        # 返回交并比overlaps -> size(1,anchor数量,gt_boxes数量)
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        '''
        overlaps
        tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                 [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
                 ...,
                 [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]], device='cuda:0')
        '''

        # 对overlaps的第2维度进行取最大
        # max_overlaps      每个anchor对gtbbox的最大交并比 -> size(1,anchor数量)      !!(1,17434)
        # argmax_overlaps   每个anchor对gtbbox的最大交并比的索引    -> size(1,anchor数量)
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        # 对overlaps的第1维度进行取最大
        # gt_max_overlaps      每个gtbbox对anchor的最大交并比 -> size(1,gt_boxes数量)      !!(1,50)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # RPN_CLOBBER_POSITIVES = false
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # anchor的交并比小于阈值0.3 置0 判断为背景
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # gt_max_overlaps size(1,gt_boxes数量) -> (1,1,gt_boxes数量) -> (1,1,gt_boxes数量) -> (1,anchor数量,gt_boxes数量)
        # overlaps.eq -> 得到一个大小相同的矩阵,相同的位置放1,并对维度2进行求和 -> 统计每个anchor对应的最大交并比gtbox的个数
        # keep -> 大小[1, 17434]
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            #labels 大小(batch_size=1, anchor数量inds_inside),前景anchor置为1
            labels[keep>0] = 1

        # fg label: above threshold IOU
        # 阈值>0.7,设置为前景
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # 阈值0.3 这里没用RPN_CLOBBER_POSITIVES = False 和上面重复 但意思相反???
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # 阈值0.3
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # num_fg = 0.5 * 256
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        # 统计前景和背景的个数
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        # batch_size = 1 循环中限制正负样本的个数
        for i in range(batch_size):
            # subsample positive labels if we have too many
            # 正样本数 > num_fg
            if sum_fg[i] > num_fg:
                # labels -> 大小(1, anchor数量)     labels[1] -> 大小(anchor数量)
                # fg_inds,是前景的索引号index
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                # 根据fg_inds的元素个数随机排列 -> 转化为tensor ->仿照gt_boxes的类型
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                # 把 多余num_fg的部分进行屏蔽
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            # num_bg = 256 - 正样本的个数
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            # 负样本数 > num_fg 处理方法和上面相同
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        # offset = [0] * 1
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        # argmax_overlaps -> size(1,anchor数量)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        '''
        输入:
            anchors     ->      size:(W*H*12,4)
            # gt_boxes  ->      size:(1, gt数量, 5)
            # argmax_overlaps   每个anchor对gtbbox的最大交并比的索引    -> size(1,anchor数量)
            gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)
                        ->      size:(1, anchor数量, 5)       ->  意义:每个anchor所对应的GT标签(根据IOU判断)可能是0对应背景
        输出:
            bbox_targets -> size([1, anchor个数, 4])  每个anchor与IOU最大的GTbbox的位移和缩放参数
        '''
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        # cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0] = 1.0
        # bbox_inside_weights -> 大小(batch_size=1, Anchor个数) 在前景部分甚至权重1
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        # RPN_POSITIVE_WEIGHT = -1.0
        # 设置正负样本的权重
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # i = 0 固定 ,num_examples计算前景的Anchor的个数
            # label 背景为-1
            num_examples = torch.sum(labels[i] >= 0)
            # num_examples.item() -> type(float)
            # 正样本权重
            positive_weights = 1.0 / num_examples.item()
            # 负样本权重
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        # bbox_outside_weights -> 大小(1,Anchor个数)
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # _unmap -> 进行数据的填充 -> 大小都是(1,Anchor数量) -> 在面积不为0的anchor处填入第一个数据(如labels)
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []
        # labels 大小 -> (1,Anchor数量) -> (1, 12, H, W) -> (1, 1, 12*H, W) -> 添加到output list
        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        # bbox_targets 大小 -> (1,Anchor数量) -> (1, 12*4, H, W) -> 添加到output list
        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        # bbox_inside_weights 大小 -> (1,Anchor数量) -> (1, Anchor数量, 4) -> (1, 12*4, H, W)添加到output list
        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        # bbox_outside_weights 大小 -> (1,Anchor数量) -> (1, Anchor数量, 4) -> (1, 12*4, H, W)添加到output list
        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)
        # 返回output列表[标签, bbox坐标变化, in权重, out权重]
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

# 填充数据的函数 例如  _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        # 初始化 ret -> 大小(1,anchor数量),值为-1
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        # 取有非0anchor 填入data
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
