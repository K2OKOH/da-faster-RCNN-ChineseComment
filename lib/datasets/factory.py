# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
    Factory method for easily getting imdbs by name.
    能够更简单的用名字找到数据集, 里面设置了对多种数据集的预处理
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 创建一个空表并导入5个数据集
__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.cityscape import cityscape
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)  # 如voc_2007_train
    # 使用lambda表达式的意义？？
    # __sets[]
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year)) # 如果sets[name]=pascal_voc(split, year),数据集实例化

for year in ['2007', '2012']:
  for split in ['train_s', 'train_t', 'train_all', 'test_s', 'test_t','test_all']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cityscape(split, year))


# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))
        
# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

# 如name = "cityscape_2007_train_s"
# __sets 是在import是执行并创建的字典.
def get_imdb(name):
  """Get an imdb (image database) by name."""
  """通过名称获取imdb(image database)"""
  if name not in __sets:  # __sets 刚开始为空的字典，在函数import时执行后填好字典，如果该数据集的名字不在该字典中
    raise KeyError('Unknown dataset: {}'.format(name))  # 没有就报错，未知数据集
  return __sets[name]()   # 返回字典中的该项目,项目是一个数据集标注的实例化对象


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
