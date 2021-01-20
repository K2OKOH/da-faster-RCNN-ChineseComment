from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# 生成anchor
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # base_anchor   ->  array([0, 0, 15, 15])
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    '''
    ratio_anchors !!例子,根据base_anchor形成三个框(可能有负数,)
    [   [-3.5  2.  18.5 13. ]
        [ 0.   0.  15.  15. ]
        [ 2.5 -3.  12.5 18. ]   ]
    '''
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # 上面进行ratio扩展,下面进行面积上的扩展
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    # 最后的anchor是二维数组,每行为一个框
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    # 加入新维度
    '''
    ws !!例子
    [23. 16. 11.]
    ->
    [   [23.]
        [16.]
        [11.]   ]
    '''
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # np.hstack在水平方向堆叠
    '''
    anchors !!例子,形成三个框有负数
    [   [-3.5  2.  18.5 13. ]
        [ 0.   0.  15.  15. ]
        [ 2.5 -3.  12.5 18. ]   ]
    '''
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    # 得到baseanchor的宽,高和中心坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    # 得到一组 size_ratios 是w^2
    size_ratios = size / ratios
    # 四舍五入,得到宽度 anchor 的np数组
    ws = np.round(np.sqrt(size_ratios))
    # 高度 = 宽度 * H/W (np数组)
    hs = np.round(ws * ratios)
    # 创造三个比例框的叠加 二维np
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # 取base_anchor的w,h,中心坐标,进行大小上的扩展
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
