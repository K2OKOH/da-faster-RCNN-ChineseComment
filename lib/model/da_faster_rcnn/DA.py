from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
from model.da_faster_rcnn.LabelResizeLayer import ImageLabelResizeLayer
from model.da_faster_rcnn.LabelResizeLayer import InstanceLabelResizeLayer



class GRLayer(Function):

    # 这里的ctx，其实就是self
    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1
        # 设定一个参数，input保持不变的传输
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # 梯度反向，并乘上系数
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    # 新样式，不用先实例化    !!参考 https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845
    return GRLayer.apply(x)

# 图像级对齐
class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          512*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    # x -> size([1,512,H,W])    FeatureMap
    def forward(self,x,need_backprop):
        # 梯度反转
        x=grad_reverse(x)
        # 两层卷积 维度:512 -> 512 -> 2
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        # x -> size([1, 2, H, W])
        # 根据图片数量生成label数组
        label=self.LabelResizeLayer(x,need_backprop)
        # label -> ([1, 37, 75])
        return x,label

# 实例级对齐
class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        # x -> size([256,4096])
        x=grad_reverse(x)
        # 3层全连接 维度:4096 -> 1024 -> 1024 -> 1
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        # x -> size([256,1])
        label = self.LabelResizeLayer(x, need_backprop)
        # label -> size([256,1])
        return x,label


