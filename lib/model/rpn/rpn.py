from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES  # 是[4,8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS  # 是[0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]   # 是16,vgg下采样倍数

        # define the convrelu layers processing input feature map
        # 输入,输出 -> 512, 卷积核 -> 3*3, 步长 -> 1, 补周边 -> 1,需要偏置
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 进行fg/bg判断, 4 * 3 * 2 作为分类卷积输出的维度
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)??
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 对bbox进行分类, 维度4 * 3 * 4 = 48
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        # 实例化 proposallayer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        '''
        输入:
            base_feat   ->  feature_map     ->  size(1,512,H,W)
            im_info     ->  image W,H,ratio ->  size(1,3)
            gt_boxes    ->  GroundTruthBox  ->  size(1,数量,5)
            num_boxes   ->  目标框的数量      ->  size(1)
        '''
        # base_feat的维度0,是1,base_feat是四维,但只有一张图
        # base_feat.size(), !! 例如 [1, 512, H, W] (一张图, 512维, 大小H*W), 大小视输入像素确定
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        # 进行RPN卷积后relu,  rpn_conv1 -> [1, 512, H, W]
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        # 第一路 ,进行(fg/bg)预测
        # get rpn classification score
        # 经过卷积, 得到分类得分,只分前景背景(fg/bg)??, rpn_cls_score -> [1, 24, H, W]
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        # 改边前景背景分类得分形状 [1, 24, H, W] -> [1, 2, 12*H, W]
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # 对rpn_cls_score_reshape的第一个维度进行softmax, 相当于分成两类
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        # softmax后再把形状变回来nc_score_out = 3*4*2 = 12 形状:[1, 2, 12*H, W] -> [1, 24, H, W]
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # 第二路 ,进行bbox的预测
        # get rpn offsets to the anchor boxes
        # 进行一层卷积, rpn_cls_score -> [1, 48, H, W]
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # 进行roi的预测
        '''
        形成两个支路,(1)预测fg/bg (2)进行Anchor的预测并根据fg/bg进行非极大值(nms)删选
        输入:
            是个tuple(),元素如下
            rpn_cls_prob    ->  torch.size([1, 24, H, W]), 每层的像素是fg/bg的概率
            rpn_bbox_pred   ->  预测bbox的feature_map(大小:[1, 48, H, W])
            im_info         ->  图片的大小H, W, ratio
            cfg_key         ->  'TRAIN' or 'TEST'
        输出:
            rois            ->  size([1, num_proposal, 5])
            rois是anchor经过fg/bg预测 + nms 筛选过后的proposal, num_proposal<=2000(目标域是300), 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
        '''
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        # 源域 需要进行, 目标域 跳过
        if self.training:
            # 如果 gt_boxes 不存在就警告
            assert gt_boxes is not None
            '''
            输入:
                rpn_cls_score   :FeatureMap经过一层分类的卷积后  ->  [1, 24, H, W]
                gt_boxes   ->  [1, 数量, 5]
                im_info    ->  [1,3]
                num_boxes  ->  [1]
            输出:
                返回output列表[标签, bbox坐标变化, in权重, out权重]
                    labels                  大小 -> (1, 1, 12*H, W)   : 标签(0:背景, 1:前景, -1:屏蔽)
                    bbox_targets            大小 -> (1, 12*4, H, W)   : 每个anchor与IOU最大的GTbbox的位移和缩放参数
                    bbox_inside_weights     大小 -> (1, 12*4, H, W)   : in权重
                    bbox_outside_weights    大小 -> (1, 12*4, H, W)   : out权重
            '''
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss 计算分类损失
            # rpn_cls_score前景背景分类得分     形状[1, 2, 12*H, W] -> [1, 12*H, W, 2] -> [1, 12*H*W, 2]
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            # 取出rpn_data的label 变化形状为 -> (1, 12*H*W)
            rpn_label = rpn_data[0].view(batch_size, -1)

            # 找到rpn_label中的非屏蔽项(≠-1)的索引号index 最后是一维的数组
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            # 前背景分类得分rpn_cls_score 先变形状为(12*H*W, 2) 对第0维度 寻找需要匹配的anchor
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            # 前景背景label 先变形状为(12*H*W) 对第0维度 寻找需要匹配的anchor
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            # 变成变量(新版本中就不用了)
            rpn_label = Variable(rpn_label.long())
            # 计算交叉损失(分类)
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # 计算前景的个数
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # 取出[bbox坐标变化, in权重, out权重]
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss 都变成变量(新版本中就不用了)
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            # 计算L1loss(回归)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        # 返回roi,分类和回归的损失
        return rois, self.rpn_loss_cls, self.rpn_loss_box
