# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import imageio
# from scipy.misc import imread #针对ba版本改为了imageio
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb

# 输入的是一个列表包含[一张图片的roi的字典]
def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  # 有几张图片的字典,根据输入只有一张 -> 文件名
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  # 产生np随机数组,大小是num_images(这里是1),从0到len-1,(cfg.TRAIN.SCALES)是图片短边的像素是个元组
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)

  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  # 返回图片的np数组,和缩放比
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  # 创建数组添加图片np数组作为data键值的数据
  blobs = {'data': im_blob}

  # 取图片的路径
  im_name=roidb[0]['image']
  # 在路径中没找到source,就是目标域,添加关于BP的键值
  # target_domain不需要反传BP
  # source_domain需要反传BP
  if im_name.find('source_') == -1:  # target domain
    blobs['need_backprop']=np.zeros((1,),dtype=np.float32)
  else:
    blobs['need_backprop']=np.ones((1,),dtype=np.float32)


  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  # 使用所有的GroundTruth
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    # 返回元组(其中元素是GT_class不是背景0的索引值),[0]取第一个维度(一维数组)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  # 返回一个数组size->(前景的个数,5(是gt_box的数量))不进行初始化
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  # 填入经过缩放比运算后的框的坐标
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  # 填入gt_class
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  # 在字典中添加GT标签
  blobs['gt_boxes'] = gt_boxes
  # 在字典中添加图片的信息,长,宽,缩放比
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)
  # 在字典中添加图片的ID序号
  blobs['img_id'] = roidb[0]['img_id']
  # 返回blobs字典
  return blobs

# 返回图片的np数组,和缩放比,并进行图片的缩放 和 去均值处理
def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  # 有几张图片,根据输入->只有一张
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  # 只有一张
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image']) 因为版本问题进行修改
    # 读取字典中image的键值 -> 文件的路径,读取图片
    im = imageio.imread(roidb[i]['image'])
    # 如果图像是二维(无色彩信息)
    if len(im.shape) == 2:
      # 增加了第三个维度
      im = im[:,:,np.newaxis]
      #对第三个维度进行扩展(为了程序兼容2维图像)
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    # 使im倒叙(对第三个通道),(特殊用法[i:j:s(步长)])
    # 为了兼容cv2
    im = im[:,:,::-1]

    # 如果需要反转对第二通道进行倒叙
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    # 获取短边像素
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    #(cfg.PIXEL_MEANS)是像素均值,(cfg.TRAIN.MAX_SIZE)是长边像素
    # 返回缩放后的图片和缩放比
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    # 形成缩放列表
    im_scales.append(im_scale)
    # 形成图片表
    processed_ims.append(im)
    # 其实这里列表中也就只有一个元素,这么做可能是为了兼容性??

  # Create a blob to hold the input images
  # 得到图片的np数组
  blob = im_list_to_blob(processed_ims)
  # 返回图片的np数组,和缩放比
  return blob, im_scales
