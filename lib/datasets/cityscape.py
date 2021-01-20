from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse     # 稀疏矩阵
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET  # 元素树对xml文件解析
import pickle   #持久化存储模块
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

# cityscape('train', '2007')
# 继承imdb
class cityscape(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        # 使用父类的初始化方法，仅给出数据集名称
        imdb.__init__(self, 'cityscape_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set # image_set !! 例如：'train_s', 'train_t', 'train_all', 'test_s', 'test_t','test_all'
        # 定位数据集路径
        # 如果devkit_path不存在：self._devkit_path = self._get_default_path()（默认路径）!!例如：'~\DAF\data\cityscape'
        # 如果devkit_path存在：self._devkit_path = devkit_path（开发工具箱地址）
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        # 数据的路径，后加上该_year（年份）的子文件夹 !!例如：'~\DAF\data\cityscape'+'\VOC2007'
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)

        # class的类别（元组）
        self._classes = ('__background__',  # always index 0
                         'person', 'rider', 'car',
                         'truck', 'bus', 'train',
                         'motorcycle', 'bicycle')

        # 打包做成字典{'class0':0, 'class1':1, ....}, （class to index）
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'    # （文件扩展名）
        # 把对应域和训练测试的txt文件中的图片名，->导入self._image_index列表中,每个元素代表一张图片
        self._image_index = self._load_image_set_index()     # train image name without .jpg 没有扩展名

        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        # 根据地址打开标注文件(Annoation),并以字典的形式导入缓存文件中,并返回gt_roidb字典
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())  # 生成当前版本号,每次不同->作用??
        self._comp_id = 'comp4'

        # PASCAL specific config options
        # 一些不知道是什么的配置????
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        # 如果数据集路径不存在,就在控制台上提示
        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)

        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # 针对VOC2007下的ImageSet/main文件夹，在文件夹中是.txt的文本文件，
    # txt文件用于区分'源域&目标域'和'训练&测试'（如：test_s.txt）,文件中每一行对应一张图片名
    # _load_image_set_index（）方法，用于把一个txt文件中的图片名导入到image_index列表中，并返回（图片名不带扩展名）
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt

        # _data_path & _image_set 在__init__()中定义 !! 例如：'~\DAF\data\cityscape'+'\VOC2007'
        # image_set_file路径 !! 例如：'~\DAF\data\cityscape\VOC2007\ImageSets\Main\train_s.txt'
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        # 如果路径不存在，就提示在控制台中
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        #创建空列表，用于……
        image_index=[]
        # 打开文件
        with open(image_set_file) as f:
            # 读每一条（该行>一个字符时生效）
            for x in f.readlines():
                if len(x)>1:
                    # image_index 列表添加上该行的数据，x.strip()除去首位的空格
                    image_index.append(x.strip())
            #image_index = [x.strip() for x in f.readlines()]

        # 返回image_index列表
        return image_index

    # 获取默认路径，进行地址的拼接，（cfg.DATA_DIR）定义的数据集路径+'cityscape'
    # cfg.DATA_DIR在cinfig.py中定义
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'cityscape')

    # 根据地址打开标注文件(Annoation),并以字典的形式导入缓存文件中,并返回gt_roidb字典
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # self.cache_path 在imdb中定义，是建立在数据集文件夹中的缓存路径，作用？？
        # 建立该（训/测，S/D）集合的缓冲文件路径
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # 缓冲文件，存在就打开
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                # 将二进制对象转换成 Python 对象
                roidb = pickle.load(fid)
            # 打印从哪里了gt roidb的缓冲文件
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            # 返回该gt roidb的缓冲文件
            return roidb
        # 根据image_index中的元素，从标注（Annoation）文件夹中导入每张图片的标注，返回字典
        # self.image_index中取的index是一张图片的名称
        # gt_roidb的数据格式:[{'':[[,],[,]...]},{...},{...},...],每张图返回一个字典,gt_roidb是个列表,存放字典,字典中存放对应键值的字典.
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        # 把标注(gt_roidb)写入缓存文件中
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)     # 对超大对象的支持
        print('wrote gt roidb to {}'.format(cache_file))    # 打印写入缓存文件

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    # 阿这，用的pascal数据集的导入方法却没有改名字！？
    # 返回的是字典
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        # 根据index(ImageSets文件夹中的txt文件)，找到标注文件(Annotations)的路径
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # 对xml解析（元素树）
        tree = ET.parse(filename)
        # 在元素树中找 'object' ，object就是图片中被框出的物体
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        # 判断 物体个数
        num_objs = len(objs)

        # 建立空np数组（num_objs*4）->存放GT目标框用
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # 建立空np数组（num_objs）->存放GT类别
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # 建立空np数组（num_objs（目标数量）*num_classes（类别数量））->所有obj(目标)在不同class(类别)的置信度
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # np数组->box框的面积
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # np数组->目标是否是难目标
        ishards = np.zeros((num_objs), dtype=np.int32)

        # 对每个目标分别进行读取,enumerate枚举:返回序号,元素
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')   # 找<object>标签下的<bndbox>标签
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1  # 坐标从零开始所以要减去
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')   # 找<object>标签下的<difficult>标签
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult     # 写入判断难易样本的数组

            # <name>标签下的类别信息全部小写,除去空格,再通过self._class_to_ind字典找到类别对应的编号,赋值给cls(cls是个数值临时变量)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]     # 对每个obj目标,写入GT的boxes数组
            if boxes[ix,0]>2048 or boxes[ix,1]>1024:        # 最小值超越边界检测  必要性??
                print(boxes[ix,:])
                print(filename)
                p=input()

            gt_classes[ix] = cls    # GT标签赋值
            overlaps[ix, cls] = 1.0     # 目标对每个class的置信度初始化,GT是1,其他是0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)   # GT的box的面积(像素)

        # 建立空np数组(num_objs(目标数量)*num_classes(类别数量))-> 作用不明？？？？？？
        # 通过实验并不发生改变????
        overlaps = scipy.sparse.csr_matrix(overlaps)

        # 返回读到的数据
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,   # 不翻转
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True



