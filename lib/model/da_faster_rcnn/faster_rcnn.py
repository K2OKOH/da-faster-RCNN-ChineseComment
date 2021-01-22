import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()     # 继承父类的__init__()方法
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        # 实例化,RPN网络(self.dout_base_model)是512,vgg16子类中定义,输入rpn的维度
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        # grid_size = 7 * 2 = 14
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # dout_base_model = 512
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)

    # 前向传播 (最最最最关键)
    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop):
        # 源域需要BP,目标域不需要BP时正常,否则报错
        assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0

        # 读取相关数据
        batch_size = im_data.size(0)
        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop=need_backprop.data


        # feed image data to base model to obtain base feature map
        # 在vgg16的子类中定义,是分类网络不要最后一个最大池化层,输出(512维,大小不固定(根据输入图片而定)) -> 所谓feature_map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        # 设置rpn网络在训练状态
        self.RCNN_rpn.train()
        # 把feature_map和标注的参数输入rpn网络,进行fg/bg的分类
        '''
        输入:
            base_feat   ->  feature_map     ->  size(1,512,H,W)
            im_info     ->  image W,H,ratio ->  size(1,3)
            gt_boxes    ->  GroundTruthBox  ->  size(1,数量,5)
            num_boxes   ->  目标框的数量      ->  size(1)
        输出:
            rois        ->  size([1, num_proposal, 5])
                rois是anchor经过fg/bg预测 + nms 筛选过后的proposal, num_proposal<=2000, 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
            rpn_loss_cls    ->单个值
            rpn_loss_bbox   ->单个值
        '''
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            '''
            作用:
                再次对roi进行筛选(到256个vgg16.yml中设定)
                roi对应的GT标签(之前的步骤只有fg,bg,这里得到的是class)
                roi的GT变化量(之后就是要通过这个做回归)
                得到权重
            输入:
                rois        ->  size([1, num_proposal, 5])
                    rois是anchor经过rpn预测保留的前景fg + nms 筛选过后的proposal, num_proposal<=2000, 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
                gt_boxes    ->  GroundTruthBox  ->  size(1,数量,5)
                num_boxes   ->  目标框的数量      ->  size(1)
            输出:
                rois_data   -> list
                    rois            ->  size([1,256,5])    预测框:最后一维 前1:0   后4:坐标
                    rois_label      ->  size([1,256])      正样本的标签
                    rois_target     ->  size([1,256,4])     -> 两个平移变化量，两个缩放变化量
                    rois_inside_ws  ->  size([1,256,4])     ->  最后一维度:(1.0, 1.0, 1.0, 1.0)
                    rois_outside_ws ->  size([1,256,4])     ->  最后一维度:(1.0, 1.0, 1.0, 1.0)
            '''
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # roi_pooling
        # POOLING_MODE = 'crop' 进行裁剪
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            '''
            # rois.view(-1, 5) 大小 ->  (proposal数量,5)
            # base_feat 大小        ->  (1,512,H,W)
            # self.grid_size       ->   14
            '''
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        # pooled_feat -> 大小(256,512,7,7)
        # 利用vgg16的顶层(除最后一层) 输出 4096 个值
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        # 是回归输出 4 * class 个值
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        # 进行分类
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        # 根据分类和回归计算 RCNN_loss
        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for target =========================="""
        """ =================== 对于 目标域  =========================="""
        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        # 提取featuremap
        tgt_base_feat = self.RCNN_base(tgt_im_data)

        # feed base feature map tp RPN to obtain rois
        # 设定为测试,不进行training步骤,不算损失
        self.RCNN_rpn.eval()
        # 前景背景判断,仅仅预测roi,目标域无标签不进行loss计算
        # tgt_rois  ->  size([1, num_proposal, 5])  ->  num_proposal<=300(和源域不同), 最后一维[第一个元素恒定为0,x1,y1,x2,y2]
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        #roi pooling
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size)
            tgt_grid_yx = torch.stack([tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(tgt_base_feat, Variable(tgt_grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        # feed pooled features to top model
        # 输入顶层网络
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        """  DA loss   """
        """  域对抗损失  """

        # DA LOSS
        DA_img_loss_cls = 0
        DA_ins_loss_cls = 0

        tgt_DA_img_loss_cls = 0
        tgt_DA_ins_loss_cls = 0

        # 提取源域特征
        # 这里的need_backprop??
        base_score, base_label = self.RCNN_imageDA(base_feat, need_backprop)

        # Image DA 图像级对齐
        base_prob = F.log_softmax(base_score, dim=1)
        DA_img_loss_cls = F.nll_loss(base_prob, base_label)

        # 实例级对齐
        '''
        输入roi特征,输出域特征
        输入:
            pooled_feat     ->  size([256,4096])
            need_backprop   ->  size([1])
        输出:
            instance_sigmoid->  size([256,1])
            same_size_label ->  size([256,1])
        '''
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(pooled_feat, need_backprop)
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # 一致性对齐
        #consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
        consistency_prob=torch.mean(consistency_prob)
        consistency_prob=consistency_prob.repeat(instance_sigmoid.size())

        DA_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

        """  ************** taget loss ****************  """

        tgt_base_score, tgt_base_label = \
            self.RCNN_imageDA(tgt_base_feat, tgt_need_backprop)

        # Image DA
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)


        tgt_instance_sigmoid, tgt_same_size_label = \
            self.RCNN_instanceDA(tgt_pooled_feat, tgt_need_backprop)
        tgt_instance_loss = nn.BCELoss()

        tgt_DA_ins_loss_cls = \
            tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)


        tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
        tgt_consistency_prob = torch.mean(tgt_consistency_prob)
        tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())

        tgt_DA_cst_loss = self.consistency_loss(tgt_instance_sigmoid, tgt_consistency_prob.detach())


        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
               DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls,DA_cst_loss,tgt_DA_cst_loss


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        # 初始化模型,在vgg16的子类中定义
        self._init_modules()
        # 初始化参数
        self._init_weights()
