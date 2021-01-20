
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_da_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    # class的数量
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)   #所有图片的个数

    # given the ratio_list, we want to make the ratio same for each batch.
    # 希望每个batch中图片的长宽比相同
    # 创建一个和图片个数相同的0张量
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()

    # 计算batch的个数，除法向上取整，转化为整形
    num_batch = int(np.ceil(len(ratio_index) / batch_size))
    # 对每个batch操作
    for i in range(num_batch):
        # batch的头索引(每个batch的第一张图)
        left_idx = i*batch_size
        # batch的尾索引(最后一个batch不超过总体的尾部)(每个batch的最后一张图)
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        # 这一步操作有些迷 :(  (长宽比列表是从小到大)
        #   首图    尾图    设定
        #    高     高    跟随尾图
        #    低     低    跟随首图
        #    首尾图不同     设为1

        # batch最后一张图的长宽比 < 1  (高图片)
        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        # batch最第一张图(宽图片)
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        # 有高有低的话，长宽比设置为1
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        # 每个batch设置一样的长宽比
        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio    # trainset ratio list ,each batch is same number

  # 使用[]运算时调用,类似于列表
  def __getitem__(self, index):        # only one sample
    # 如果在训练过程中
    if self.training:
        # s_ratio_list  -> 排列后的长宽比列表(从小到大)
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    '''
      根据长宽比(从小到大)取出图片对应roi参数的字典{}
      {'boxes': boxes,
       'gt_classes': gt_classes,
       'gt_ishard': ishards,
       'gt_overlaps': overlaps,
       'flipped': False,  # 不翻转
       'seg_areas': seg_areas}
    '''
    # minibatch_db 是列表[]->里面是一张图片的roi字典
    minibatch_db = [self._roidb[index_ratio]]
    '''
    # 关键->得到blobs字典包含
    'data':图片（四维np）但其实只有一张图片的三维
    'need_backprop':一维np数组[.1]要BP
    'gt_boxes':Reg+cls用，二维np数组,每个目标有一个[]
    'im_info':二维np数组,图像的ID和序号,但只有1张图
    'img_id':int图片序号
    '''
    blobs = get_minibatch(minibatch_db, self._num_classes)
    # 把数据读入torch的变量中
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    # 取图片的H和W
    data_height, data_width = data.size(1), data.size(2)
    # 如果在训练阶段
    if self.training:
        """
            da-faster-rcnn layer............
        """
        # 打乱bbox的顺序,并转移到torch
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        need_backprop=blobs['need_backprop'][0]



        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        # 读入一个batch的目标长宽比
        ratio = self.ratio_list_batch[index]

        # 进行图片的裁剪(如果需要),data裁剪 + gt_boxes坐标改变
        if self._roidb[index_ratio]['need_crop']:
            # 如果是高图片
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                # 读取bbox的最高点和最低点
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                # 长边height需要裁剪成为的大小
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height
                # bbox的最大距离
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    # bbox的最大距离 < 裁剪范围
                    if (box_region-trim_size) < 0:
                        # 设点裁剪最低点的范围,并在范围中随机选择
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    # bbox的最大距离 >= 裁剪范围
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        # 刚好相等
                        if y_s_add == 0:
                            y_s = min_y
                        # bbox的最大距离 > 裁剪范围
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                # 进行裁剪,按照以上原则,保证长宽比确定,->尽可能多的包含bbox的面积
                data = data[:, y_s:(y_s + trim_size), :, :]

                # bbox的坐标跟随着裁剪进行变更
                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                # 防止超出图片的边界(bbox的最大距离 > 裁剪范围)的情况下
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)
            # 如果是宽图片,类似操作
            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        # 进行图像的拉伸
        # 高图片
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            # 创建一个矩阵(高*宽*3),但是之前不是裁剪过了??
            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()
            # 有什么区别?? data_height 和 np.ceil(data_width / ratio)
            padding_data[:data_height, :, :] = data[0]
            # update im_info
            # 更改图片信息
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        # 宽图片
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # check the bounding box:
        # 选出有面积的bbox，形成列表
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        # 创建数组(bbox的数量 * 维度(5)),初始化未0
        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        # 如果keep张量的元素个数不为0
        if keep.numel() != 0:
            # 取出bbox的值
            gt_boxes = gt_boxes[keep]
            # 取出bbox的数量
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            # 写入张量中
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        # 进行维度转化,通道数放在最前
        # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)
        '''
        # 返回的是什么
        # padding_data 图像数据
        # im_info
        # gt_boxes_padding -> bbox的5个标注
        # num_boxes -> bbox的数量
        # need_backprop -> 是否需要反向传播
        '''
        return padding_data, im_info, gt_boxes_padding,num_boxes,\
               need_backprop
    # 不是训练过程 -> 并不加载GT
    else:

        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0
        need_backprop=0

        return data, im_info, gt_boxes, num_boxes,need_backprop

  def __len__(self):
    return len(self._roidb)
