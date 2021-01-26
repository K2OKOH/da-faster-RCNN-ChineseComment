# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader

from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roibatchLoader import roibatchLoader


from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.da_faster_rcnn.vgg16 import vgg16
from model.da_faster_rcnn.resnet import resnet

#from model.da_faster_rcnn.vgg16 import vgg16
#from model.da_faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='cityscape', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="/data/ztc/adaptation/Experiment/da_model",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.002, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=6, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--lamda', dest='lamda',
                      help='DA loss param',
                      default=0.1, type=float)


# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args

# 采样器,继承自pytorch模块
class sampler(Sampler):
  # 初始化 参数(训练集,batch的大小) -> 图片个数
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)   # batch的个数
    self.batch_size = batch_size
    # arange(0,batch_size)整形,不包含batch_size -> view(1, batch_size)从1到batch_size ->long()64位整形
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False  # 没有图片剩余
    if train_size % batch_size: # 有图片剩余
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  # 迭代器
  def __iter__(self):
    # randperm(n)返回0~n-1的随机序列 -> 随机生成batch个数的列向量 * batch_size(广播)
    rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
    # expand扩张到(batch个数*大小) -> 使得rand_num的每一行是一组banch的连续序号
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
    # 变成一维的向量
    self.rand_num_view = self.rand_num.view(-1)

    # 有余数把剩余的加上
    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    # 返回一个迭代器（取图片的顺序）
    return iter(self.rand_num_view)


  # 返回数据集图片的个数
  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  # 读入数据集，设置名称和格式
  if args.dataset == "pascal_voc":
      print('loading our dataset...........')
      args.imdb_name = "voc_2007_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_t"
      args.s_imdbtest_name="cityscape_2007_test_s"
      args.t_imdbtest_name="cityscape_2007_test_t"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  # 载入设置文件的地址
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  # 设置文件地址不为空
  # 读取网络设定的参数(如vgg16.yml)
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # 打印设置文件
  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  # 有cuda显卡但是没用，提醒应该用了
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  # 对roidb进行融合
  # args.s_imdb_name="cityscape_2007_train_s"
  # combined_roidb()方法在roidb.py文件中
  '''
      s_imdb        -> 实例化后的数据集       !! 例如 imdb = cityscape(train_s, 2007)
      s_roidb       -> 每张图片标注字典的列表  !! 例如 [{ 第一张图片的字典 },{ 第二张图片的字典 },{...}]
      s_ratio_list  -> 排列后的长宽比列表
      s_ratio_index -> 长宽比的次序
  '''
  s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)   # 源域
  s_train_size = len(s_roidb)  # add flipped         image_index*2    # 源域图片个数

  t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)   # 目标域
  t_train_size = len(t_roidb)  # add flipped         image_index*2    # 目标域图片个数

  print('source {:d} target {:d} roidb entries'.format(len(s_roidb),len(t_roidb)))

  # 输出路径,没有路径就创建
  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # batch采样器的实例化
  s_sampler_batch = sampler(s_train_size, args.batch_size)
  t_sampler_batch = sampler(t_train_size, args.batch_size)

  # roibatchloader实例化
  s_dataset = roibatchLoader(s_roidb, s_ratio_list, s_ratio_index, args.batch_size, \
                           s_imdb.num_classes, training=True)
  # dataloader实例化
  s_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=args.batch_size,
                            sampler=s_sampler_batch, num_workers=args.num_workers)


  t_dataset=roibatchLoader(t_roidb, t_ratio_list, t_ratio_index, args.batch_size, \
                           t_imdb.num_classes, training=False)

  t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=args.batch_size,
                                           sampler=t_sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  # 初始化一些要用到的变量
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  need_backprop = torch.FloatTensor(1)

  tgt_im_data = torch.FloatTensor(1)
  tgt_im_info = torch.FloatTensor(1)
  tgt_num_boxes = torch.LongTensor(1)
  tgt_gt_boxes = torch.FloatTensor(1)
  tgt_need_backprop = torch.FloatTensor(1)


  # ship to cuda
  # 把上面的变量转移到GPU上
  if args.cuda:
      im_data = im_data.cuda()
      im_info = im_info.cuda()
      num_boxes = num_boxes.cuda()
      gt_boxes = gt_boxes.cuda()
      need_backprop = need_backprop.cuda()

      tgt_im_data = tgt_im_data.cuda()
      tgt_im_info = tgt_im_info.cuda()
      tgt_num_boxes = tgt_num_boxes.cuda()
      tgt_gt_boxes = tgt_gt_boxes.cuda()
      tgt_need_backprop = tgt_need_backprop.cuda()

  # make variable，把tensor设置为Variable，新版本中已经合并
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  need_backprop = Variable(need_backprop)

  tgt_im_data = Variable(tgt_im_data)
  tgt_im_info = Variable(tgt_im_info)
  tgt_num_boxes = Variable(tgt_num_boxes)
  tgt_gt_boxes = Variable(tgt_gt_boxes)
  tgt_need_backprop = Variable(tgt_need_backprop)

  # 判断是否使用cuda
  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  # 初始化网络，注意:网络的预训练模型在网络的__init__方法中定义
  if args.net == 'vgg16':
    fasterRCNN = vgg16(s_imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    # 交互式调试使用pdb
    pdb.set_trace()

  # 构建网络
  fasterRCNN.create_architecture()

  # 学习率
  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  # 创建空列表->用于存放参数
  params = []
  # 读取网络的参数名称+值
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        # 如果带有偏置b,在参数列表中添加字典
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  # 根据参数选择优化方法
  if args.optimizer == "adam":
    #lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  # 是否回复checkpoints,(中途崩溃使用??)
  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  # 是否使用多GPU训练
  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)
  # 是否使用GPU(cuda)
  if args.cuda:
    fasterRCNN.cuda()

  # 计算每个epoach的batch的个数
  iters_per_epoch = int(s_train_size / args.batch_size)

  # 展示可视化训练
  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  # 开始训练(epoch)(从开始的epoch,到最后)
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode

    # 设置训练的标志位
    fasterRCNN.train()
    loss_temp = 0
    # 记录开始时间
    start = time.time()
    # 学习率下降,每间隔lr_decay_step个epoch下调学习率
    if epoch % (args.lr_decay_step + 1) == 0:
        # 更改学习率
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        # 记录更改的学习率
        lr *= args.lr_decay_gamma

    # 生成迭代对象
    data_iter = iter(s_dataloader)
    tgt_data_iter=iter(t_dataloader)

    # 分为batch训练
    for step in range(iters_per_epoch):
      # 取得一个batch的数据
      data = next(data_iter)
      tgt_data=next(tgt_data_iter)

      '''
      data是个list，
      list包含5个tensor分别是:
        # padding_data 图像数据
        # im_info -> 图片的H,W,ratio
        # gt_boxes_padding -> bbox的5个标注(cls, reg)
        # num_boxes -> bbox的数量
        # need_backprop -> 是否需要反向传播(其实是源域目标域的标注)
      '''
      # 把variable的data改变维度并复制数据
      im_data.data.resize_(data[0].size()).copy_(data[0])  # change holder size
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      need_backprop.data.resize_(data[4].size()).copy_(data[4])

      tgt_im_data.data.resize_(tgt_data[0].size()).copy_(tgt_data[0])  # change holder size
      tgt_im_info.data.resize_(tgt_data[1].size()).copy_(tgt_data[1])
      tgt_gt_boxes.data.resize_(tgt_data[2].size()).copy_(tgt_data[2])
      tgt_num_boxes.data.resize_(tgt_data[3].size()).copy_(tgt_data[3])
      tgt_need_backprop.data.resize_(tgt_data[4].size()).copy_(tgt_data[4])



      """   faster-rcnn loss + DA loss for source and   DA loss for target    """
      # 网络梯度清零
      fasterRCNN.zero_grad()
      # 投喂参数并得到结果
      '''
      构建网络并前向传播
      输入:
            ### 源域 ###
        im_data             ->  图像数据
        im_info             ->  图片的H,W,ratio
        gt_boxes            ->  bbox的5个标注(cls, reg)
        num_boxes           ->  bbox的数量
        need_backprop       ->  是否需要反向传播(其实是源域目标域的标注)
            ### 目标域 ###
        tgt_im_data
        tgt_im_info
        tgt_gt_boxes
        tgt_num_boxes
        tgt_need_backprop
      输出:
        rois                ->  size([1,256,5])     预测框:最后一维 前1:0   后4:坐标
        cls_prob            ->  size([1, 256, 9])   预测类别:onehot,softmax后
        bbox_pred           ->  size([1, 256, 4])   预测框的坐标值(rois回归后)
        rpn_loss_cls        ->  单个值               RPN分类损失
        rpn_loss_box        ->  单个值               RPN回归损失
        RCNN_loss_cls       ->  单个值               分类损失
        RCNN_loss_bbox      ->  单个值               回归损失
        rois_label          ->  size([256])         正样本标签
        DA_img_loss_cls     ->  单个值               image-level源域损失
        DA_ins_loss_cls     ->  单个值               instance-level源域损失
        tgt_DA_img_loss_cls ->  单个值               image-level目标域损失
        tgt_DA_ins_loss_cls ->  单个值               instance-level目标域损失
        DA_cst_loss         ->  单个值               一致性源域损失
        tgt_DA_cst_loss     ->  单个值               一致性目标域损失
      '''
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label,DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls,\
      DA_cst_loss,tgt_DA_cst_loss=\
          fasterRCNN(im_data, im_info, gt_boxes, num_boxes,need_backprop,
                     tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop)

      # 计算loss
      # (.mean()加不加无所谓，因为都是一个数的tensor)
      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()\
             +args.lamda*(DA_img_loss_cls.mean()+DA_ins_loss_cls.mean()\
             +tgt_DA_img_loss_cls.mean()+tgt_DA_ins_loss_cls.mean()+DA_cst_loss.mean()+tgt_DA_cst_loss.mean())
      loss_temp += loss.item()

      # backward 反向传播,梯度清零
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      # 更新参数
      optimizer.step()

      # batch,间隔显示
      if step % args.disp_interval == 0:
        # 记录当前时间
        end = time.time()
        if step > 0:
          # 本间隔的平均loss
          loss_temp /= (args.disp_interval + 1)

        # 是否多GPU,并计算各部分的loss
        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          loss_DA_img_cls=args.lamda*(DA_img_loss_cls.item()+tgt_DA_img_loss_cls.item())/2
          loss_DA_ins_cls = args.lamda * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item()) / 2
          loss_DA_cst = args.lamda * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        # 显示当前进度,当前间隔的平均loss,学习率
        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        # 显示本间隔前景和背景比,和本间隔的花费时间
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        # 显示各个部分的loss
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f,\n\t\t\timg_loss %.4f,ins_loss %.4f,,cst_loss %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,loss_DA_img_cls,loss_DA_ins_cls,loss_DA_cst))
        # 如果使用可视化进程的话添加参数
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        # 清零累计loss
        loss_temp = 0
        # 时间重新开始
        start = time.time()

    # 如果训练满epoch
    if epoch==args.max_epochs:
        # 设置保存路径和名称
        save_name = os.path.join(output_dir, 'cityscape_consist_default.pth'.format(args.session, epoch, step))
        # 保存模型和其他参数
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        # 打印已经保存!
        print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
