from __future__ import absolute_import

import numpy as np
import torch

def nms_cpu(dets, thresh):
    # 转化为numpy
    dets = dets.numpy()
    # 出去各个参数 -> size(12000)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算面积 -> size(12000)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 降序排序,返回的是index索引号 -> size(12000)
    order = scores.argsort()[::-1]

    keep = []
    # 循环到把order比较完
    while order.size > 0:
        # 取fg的置信度最高的框
        i = order.item(0)
        keep.append(i)
        # 第0个元素和 其他元素比较,取最大值
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])#这里写错了minimum(用cpu要记得改)
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h   # 计算交的面积
        # 计算交并比
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 寻找交并比<=阈值的索引值
        inds = np.where(ovr <= thresh)[0]
        # 对置信度<=0.7的,形成新的order
        order = order[inds + 1]
    # 返回的是index索引号
    return torch.IntTensor(keep)


