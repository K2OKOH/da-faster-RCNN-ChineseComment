# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
# 注意这里引用的是da文件夹中的farster_rcnn
from model.da_faster_rcnn.faster_rcnn import _fasterRCNN
import pdb

# vgg16 继承自_fasterRCNN
class vgg16(_fasterRCNN):
  # class是roidb中的类 -> 如'car'
  # pretrained -> 不预先训练 ??
  # class_agnostic -> 没有不可知类
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    # 预训练模型路径,已经改为相对路径
    # self.model_path = '/data/ztc/detectionModel/vgg16_caffe.pth'
    self.model_path = './data/ztc/detectionModel/vgg16_caffe.pth'
    # RPN网络中使用,输入rpn的feature_map的维度
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    # 调用了_fasterRCNN的__init__()方法
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    # 实例化VGG16 网络,在pytorch的包中,没有参数 => 默认分1000类(后面删了最后一层)
    vgg = models.vgg16()
    # 如果使用预训练模型,就加载
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        # 读取预训练模型,仅读取参数
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    # 把分类用的全连接层，按顺序排列，不要最后那一个分类的层 -> 输出是4096维向量
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])    # list前加*，把list分解成独立的参数传入

    # not using the last maxpool layer. 分类网络不要最后一个最大池化层,因为不需要跟着分类输出,所以输出大小可以不是(14*14),但是512维的
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    # 固定conv3 之前的网络参数
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters():
        p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
    # 设置分类用的顶层
    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    # 全连接层,进行分类输出4096(其实没有要真正vgg16的分类层)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

