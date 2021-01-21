"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

# 由imdb得到roidb
def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  通过添加一些对训练有用的派生量来丰富imdb的roidd。
  此函数预先计算每个ROI和每个GT框之间在地面真值框上获得的最大重叠量。
  还记录了具有最大重叠的类。
  增加height...标签
  """
  # roidb 是 cityscape.py中的gt_roidb,标注信息,从annotation文件中读取,是列表,表中是字典字典结构如下

    # 'boxes': boxes,
    # 'gt_classes': gt_classes,
    # 'gt_ishard': ishards,
    # 'gt_overlaps': overlaps,
    # 'flipped': False,   # 不翻转
    # 'seg_areas': seg_areas

  roidb = imdb.roidb
  # 如果不是coco数据集的话
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]   # 打开每一张图片获取图片大小
  # 对全部的图片进行遍历
  for i in range(len(imdb.image_index)):
    # 对第i张图片,ID和路径,长度和宽度,对roidb中的字典进行添加
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    # 不是coco数据集就执行
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]   # 添加宽度信息
      roidb[i]['height'] = sizes[i][1]  # 添加高度信息
    # need gt_overlaps as a dense array for argmax
    # 每个目标对类的置信度
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()

    # max overlap with gt over classes (columns)
    # 每个目标(行)最大的置信度
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    # 每个目标(行)最大的置信度的那类
    max_classes = gt_overlaps.argmax(axis=1)
    # 添加键值
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    # np.where(max_overlaps == 0)-->找到数组中0的位置,返回二维数组,组成坐标,[0]只取第一维
    zero_inds = np.where(max_overlaps == 0)[0]  # 列出有置信度存在0的行(目标)
    assert all(max_classes[zero_inds] == 0)   # 类别是背景的找出背景
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)    #找出前景


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    # 设置长宽比的极限
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    # i 代表一张图片
    for i in range(len(roidb)):
      width = roidb[i]['width']   # 取出第 i 个图像的字典 -> 'width'键值对应的矩阵
      height = roidb[i]['height']
      ratio = width / float(height)   # 计算每个框的长宽比 -> 得到一张图片的长宽比矩阵

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0   # 超过极限的进行裁剪标记,添加到字典中

      ratio_list.append(ratio)      # 把修改后的长宽比,加入列表[把所有图片的所有框的长宽比放在一张表中]

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)  # 进行长宽比排列
    return ratio_list[ratio_index], ratio_index   # 返回排列后的长宽比,和排列顺序(从小到大)

# 过滤没有目标框的目标
def filter_roidb(roidb):
    # filter the image without bounding box.过滤没有目标框的目标
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    # 若图片中的目标个数大于i
    while i < len(roidb):
      # 没有目标框
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]    # 从列表中删除
        i -= 1    # 为了让计数器不变而-1
      i += 1  # 判断下一项

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

# 把用到的数据集的db进行融合
def combined_roidb(imdb_names, training=True):   #dataset name
  """
  Combine multiple roidbs 融合多个roidbs
  """
  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    # 如果使用翻转,数据增广 2975张 -> 5950张
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()    #  data augment
      print('done')

    print('Preparing training data...')

    # 准备imdb
    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb

  # 如imdb_name="cityscape_2007_train_s"
  def get_roidb(imdb_name):
    # get_imdb 在 factory.py中定义,通过名称获取imdb(image database)
    # imdb 是数据集标注的实例化对象( !! 例如 imdb = cityscape(train_s, 2007))
    imdb = get_imdb(imdb_name)     # return a pascal_voc dataset object     get_imdb is from factory which contain all legal dataset object

    print('Loaded dataset `{:s}` for training'.format(imdb.name))   # 打印数据集的名字 !! 例如 'cityscape_2007_train_s'
    # cfg.TRAIN.PROPOSAL_METHOD = 'gt' --> 训练方法??
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    # 由imdb变为roidb
    roidb = get_training_roidb(imdb)
    return roidb

  # 对字符串进行分割，有的数据集中是多个数据集名用‘+’相连，先分开处理。
  # 最终返回GT的roidbs,形式[ 第一种数据集->[{ 第一张图片的字典 },{ 第二张图片的字典 },{...}],第二种数据集-> [{},...],[...]]
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]

  roidb = roidbs[0]   # 这里因为只有一个数据集,即cityscapes

  # 如果数据集的个数 > 1
  if len(roidbs) > 1:
    # r是每个roidb列表
    for r in roidbs[1:]:
      # 在第一个数据集的列表中追加(后面数据集的图片标注)字典
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])  # 对第一个数据集????
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)  # 数据集合并
  else:
    imdb = get_imdb(imdb_names)

  # 如果是在训练过程
  if training:
    # 过滤没有目标框的目标    !!对cityscape : 5950张 -> 5932张
    roidb = filter_roidb(roidb)    # filter samples without bbox

  ratio_list, ratio_index = rank_roidb_ratio(roidb)   #  进行长宽比的排列,排列后的长宽比列表ratio_list & 长宽比的次序ratio_index

  return imdb, roidb, ratio_list, ratio_index  # dataset, roidb dict,ratio_list(0.5,0.5,0.5......2,2,2,), ratio_increase_index(4518,6421,.....)
