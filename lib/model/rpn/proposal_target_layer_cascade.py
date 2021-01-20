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
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)       # (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)         # (0.1, 0.1, 0.2, 0.2)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)         # (1.0, 1.0, 1.0, 1.0)

    def forward(self, all_rois, gt_boxes, num_boxes):
        '''
        输入:
            all_rois    ->  size([1, num_proposal, 5])
                all_rois是anchor经过fg/bg预测 + nms 筛选过后的proposal, num_proposal<=2000, 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
            gt_boxes    ->  GroundTruthBox  ->  size(1,GT数量,5)
            num_boxes   ->  目标框的数量      ->  size(1)
        输出:
            rois_data
        '''
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # gt_boxes_append 大小(1,GT数量,5)
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        # 第一个元素保持为0,和 all_rois 保持一致
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        # 把 all_rois(前景框) 和 gt_boxes_append(gt框) 拼接 (1,GT数量,5) + (1,proposal数量,5) -> (1,proposal + GT数量,5)
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        # rois_per_image = BATCH_SIZE = 128
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        # FG_FRACTION = 0.25
        # fg_rois_per_image = 0.25 * 128 = 32
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        # 每张图片至少一个roi
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        '''
        输入:
            all_rois    ->  size(1, proposal + GT数量, 5)
                rois是anchor经过rpn预测保留的前景fg + nms 筛选过后的proposal, num_proposal<=2000, 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
            gt_boxes    ->  GroundTruthBox  ->  size(1,GT数量,5)
            fg_rois_per_image   ->  每张图片的roi      ->  int(32)
            self._num_classes   ->  类别的数量         ->  int(10)
        输出:
            labels
            rois
            bbox_targets
            bbox_inside_weights
        '''
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        '''
        输入:
            all_rois    ->  size(1,proposal + GT数量,5)
                rois是anchor经过rpn预测保留的前景fg + nms 筛选过后的proposal, num_proposal<=2000, 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
            gt_boxes    ->  GroundTruthBox  ->  size(1,GT数量,5)
            fg_rois_per_image   ->  每张图片的roi      ->  int(32)
            self._num_classes   ->  类别的数量         ->  int(10)
        '''
        # overlaps: (rois x gt_boxes)
        # 计算交并比 返回 -> size(1,all_rois数量,gt_boxes数量)
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        # 每个rois对应的最大gt_bbox的交并比
        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        # batch_size
        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        # offset = tensor([0]) * num_proposal(proposal数量)
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        # offset 变为-> 大小(1,1) + 大小(1,2050) -> 大小(1,2050)  对于一维度向量的广播操作
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # offset 是 每个rois对应的最大交并比的gt_bbox的标号
        # 根据交并比把gt_bbox的cls的标签分配给每个proposal
        # labels 大小 -> (1,2050), 每个proposal的labels
        labels = gt_boxes[:,:,4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)

        # labels_batch大小 -> (1,32)
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        # rois_batch -> (1,32,5)
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # rois_batch -> (1,32,5)
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            # 根据交并比判断fg/bg, FG_THRESH = 0.5 选择交并比>=0.5的框, 计算符合的proposal数量
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            # BG_THRESH_HI = 0.5 BG_THRESH_LO = 0.1
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            # 既有前景fg又有背景bg, fg多于设定值32,就随机抽取
            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault. 
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error. 
                # We use numpy rand instead. 
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            # 全是前景fg
            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            # 全是背景bg
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
                
            # The indices that we're selecting (both fg and bg)
            # 拼合成为要保留的部分的索引
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            # 进行标签的保留
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            # 背景设置为0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        '''
        输出:
            labels_batch        ->  (1,32)      正样本的标签
            rois_batch          ->  (1,32,5)    预测框
            bbox_targets        ->  (1,32,5)
            bbox_inside_weights ->  
        '''
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
