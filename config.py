# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# config.py
import os
import os.path as osp
import time
from easydict import EasyDict as edict

# init
__C = edict()
cfg = __C
__C.EXP = edict()
__C.DATA = edict()
__C.TRAIN = edict()
__C.TEST =edict()

# train*************************train
__C.TRAIN.LR = 1e-5
__C.TRAIN.WEIGHT_DECAY = 5e-2
__C.TRAIN.MAX_EPOCH = 10
# 每个buffer 训练的epoch数量
__C.TRAIN.INER_EPOCH = 10
# GPU 设置
__C.TRAIN.USE_GPU = True
if __C.TRAIN.USE_GPU:
    __C.TRAIN.GPU_ID = [1,3]
    __C.MAIN_GPU = [1]
# 断点续训
__C.TRAIN.RESUME = False
__C.TRAIN.RESUME_PATH = '/data/haojie/TRANCOS_DQLL/exp/23-04-01-02-20_TRANCOS/EP_5_HitRat0.49515.pth'

# test
__C.TEST.BS = 1

# data*************************data
__C.DATA.NAME = 'UADETRAC'  # SelfLane/TuSimpleLane
if __C.DATA.NAME == 'TuSimpleLane':
    __C.DATAROOT = r'/data2/haojie/TRANCOS/'
    __C.DATA.TRAIN_LIST = osp.join(__C.DATAROOT,'train_DRL_list.json')
    __C.DATA.VAL_LIST = osp.join(__C.DATAROOT,'test_DRL_list.json')
    __C.DATA.ROOT = osp.join(__C.DATAROOT,'MyTuSimpleLane')
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH = osp.join(__C.DATAROOT,r'meanImgTemp.npy')
if __C.DATA.NAME == 'SelfLane':
    __C.DATA.TRAIN_LIST =''#  TODO:
    __C.DATA.VAL_LIST = ''  #  TODO:
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH =r''#TODO:
if __C.DATA.NAME == 'TRANCOS':
    __C.DATAROOT = r'/data2/haojie/TRANCOS/'
    __C.DATA.TRAIN_LIST =''#  TODO:
    __C.DATA.VAL_LIST = ''  #  TODO:
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH =r''#TODO:
    __C.DATA.MEAN_STD = ([0.3908707, 0.3613535, 0.36716083], [0.22240958, 0.21731742, 0.21530356])
if __C.DATA.NAME == 'UADETRAC':
    __C.DATAROOT = r'/data2/haojie/UADETRAC/'
    __C.DATA.TRAIN_LIST =''#  TODO:
    __C.DATA.VAL_LIST = ''  #  TODO:
    # meanImagePath
    __C.DATA.MEAN_IMG_PATH =r''#TODO:
    __C.DATA.MEAN_STD = ([0.3908707, 0.3613535, 0.36716083], [0.22240958, 0.21731742, 0.21530356])
# buffer 的dataloader的设置
__C.DATA.NUM_WORKS = 8
__C.DATA.BS = 4
__C.DATA.SHUFFLE = True
# img dataloder的设置
__C.DATA.TRAIN_IMGBS = 16  #  TODO:
__C.DATA.VAL_IMGBS =1#  TODO:
__C.DATA.IMGSHUFFLE = True


# DQL*************************DQL
# 最大步数
__C.MAX_STEP = 40
# 距离阈值
__C.DST_THR = 0.98
# action 数量 确定为4
__C.ACT_NUM = 3
# History数量
__C.HIS_NUM = 12
# epsilon
__C.EPSILON = 0.05
# gamma
__C.GAMMA = 0.90
# landmark 数量
__C.LANDMARK_NUM = 1
# reward
__C.reward_terminal_action = 3
__C.reward_movement_action = 5
__C.reward_invalid_movement_action = -5
__C.reward_remove_action = 1
# buffer capacity
__C.BUFFER_CAP = 1280


# exp*************************exp
__C.SEED = 233
__C.EXP.ROOT = 'exp'
now = time.strftime("%y-%m-%d-%H-%M", time.localtime())
__C.EXP.NAME = now+'_'+__C.DATA.NAME
__C.EXP.PATH = os.path.join(__C.EXP.ROOT,__C.EXP.NAME)