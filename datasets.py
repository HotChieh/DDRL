# coding=utf-8
# ----tuzixini----
# MACOS Python3.6.6
'''
载入 self_lane数据集
'''
import torch
import pdb
import collections
from torch.utils import data
from scipy import io as sio
from torch.utils.data import DataLoader
import os.path as osp
import json
import numpy as np
from PIL import Image
import os
import utils as own_transforms
import torchvision.transforms as standard_transforms
from config import cfg
import json
import shutil

class UADETRAC(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None, depth_transform=None, restore_transform=None):
        self.data_path = data_path
        self.mode = mode
        self.gt_box, self.data_files = self.get_box()
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.depth_transform = depth_transform
        # self.max_obj = cfg.MAX_BBOX
        # self.down_ratio = cfg.DOWN_RATIO
        
        # self.gt_box, self.ign_box = self.get_box()
        # self.random_rotaion = RandomRotation()
        self.guass_kernel = self.GaussianKernel(shape=(25, 25), sigma=5)
        self.restore_transform = restore_transform

    def Addkernel(self, center_x,center_y,guass_kernel,center_den_IPM):
        h, w = center_den_IPM.shape
        for z in range(len(center_x)):
            cut_x1, cut_x2,cut_y1,cut_y2 = 0, 0, 0, 0
            x, y = center_x[z],center_y[z]
            x1, y1, x2, y2 = x-12,y-12, x+13, y+13
            if x1<0:
                cut_x1 = 0-x1            
                x1 = 0
            if y1<0:
                cut_y1 = 0-y1            
                y1 = 0
            if x2 > w-1:
                cut_x2 = x2-w+1            
                x2 = w-1
            if y2> h-1:
                cut_y2 = y2-h+1            
                y2 = h-1

            a = center_den_IPM[y1:y2, x1:x2]
            b = guass_kernel[cut_y1:25-cut_y2,cut_x1:25-cut_x2]
            center_den_IPM[y1:y2, x1:x2]+=guass_kernel[cut_y1:25-cut_y2,cut_x1:25-cut_x2]
        return center_den_IPM
    def GaussianKernel(self, shape=(15, 15), sigma=0.5):
        """
        2D gaussian kernel which is equal to MATLAB's fspecial('gaussian',[shape],[sigma])
        """
        radius_x, radius_y = [(radius-1.)/2. for radius in shape]
        y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
        h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))  # (25,25),max()=1~h[12][12]
        h[h < (np.finfo(h.dtype).eps*h.max())] = 0
        max = h.max()
        min = h.min()
        h = (h-min)/(max-min)
        # sumh = h.sum()
        # if sumh != 0:
        #     h /= sumh
        return h

    def __getitem__(self, index):
        # print(index)
        fname = self.data_files[index]
        img, gt = self.read_image_and_gt(fname)

        # bbox = torch.tensor(bbox)
        # ann = torch.tensor(ann)
        if self.main_transform is not None:
            img, gt = self.main_transform(img, gt)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, gt

    def __len__(self):
        return len(self.data_files)

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.data_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        scene = fname.split('/')[0]
        json_name = fname.split('/')[-1].split('.')[0]+'.json'
        with open(os.path.join(self.data_path,scene, json_name)) as myfile:
            data=myfile.read()
        gt = json.loads(data)
        gt = gt['shapes'][0]['points'][0]
        gt = [gt[0]-img.width/2, gt[1]-img.height]
        return img, gt


    def get_box(self,):
        data_files = []
        if self.mode=='train':
            gt_file_path = os.path.join(self.data_path, '../', 'train_gt_final.txt')
            # gt_ignore_path = os.path.join(self.data_path, '../', 'train_20011_ign.txt')
            gt_txt = open(gt_file_path)
            # ign_txt = open(gt_ignore_path)
        else:
            gt_file_path = os.path.join(self.data_path, '../', 'test_gt_final.txt')
            # gt_ignore_path = os.path.join(self.data_path, '../', 'test_20011_ign.txt')
            gt_txt = open(gt_file_path)
            # ign_txt = open(gt_ignore_path)
        gt_lines = gt_txt.readlines()
        # ign_lines = ign_txt.readlines()
        gt_dict = {}
        # ign_dict = {}
        for gt_line in gt_lines:
            gt_line = gt_line.split(' ')
            # gt_line_box = []
            gt_line_box = [float(x) for x in gt_line[1:-1]]
            gt_line_box = np.array(gt_line_box).reshape(-1, 4)
            gt_dict[gt_line[0]]=gt_line_box
            data_files.append(gt_line[0])
            

        return gt_dict, data_files

    def get_num_samples(self):
        return self.num_samples



class TRANCOS(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None, depth_transform=None):
        self.data_path = data_path
        self.data_files = [filename for filename in os.listdir(self.data_path) \
                           if os.path.isfile(os.path.join(self.data_path, filename)) and filename.split('.')[-1]=='jpg' and 'ipm' not in filename.split('.')[0]]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.depth_transform = depth_transform
        # self.max_obj = cfg_data.MAX_BBOX
        # self.down_ratio = cfg_data.DOWN_RATIO
        self.mode = mode
        # self.random_rotaion = RandomRotation()

    def __getitem__(self, index):
        # print(index)
        fname = self.data_files[index]
        img, vp = self.read_image_and_gt(fname)

        # bbox = torch.tensor(bbox)
        # ann = torch.tensor(ann)
        if self.main_transform is not None:
            img, vp = self.main_transform(img, vp)
        if self.img_transform is not None:
            img = self.img_transform(img)
            # ipm = self.img_transform(ipm)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
            index1 = depth < 0
            index2 = depth > 20000
            depth[index1] = 0
            depth[index2] = 0
            depth /= 20000
        # if cfg_data.CROP==True and self.mode=='train':
        #     img, den, depth, bbox, ann = self.random_crop(img, den, depth, bbox, ann, fname)
            # print('ann, bbox', len(ann), len(bbox))
        # img_h, img_w = img.shape[1], img.shape[2]
        # output_w = img_w * self.down_ratio
        # target_wh, reg_mask, ind, ann = self.bbox_transform(bbox, self.max_obj, output_w, ann)
        # print(depth.shape[1])
        # if depth.shape[1]==1:
        # print(fname)
        # if self.mode=='test':
        #     print('img_size',img.shape)
        return img, vp

    def __len__(self):
        return len(self.data_files)

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.data_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        fname = fname.split('.')[0]
        vp = open(os.path.join(self.data_path, fname+'vl.txt'))
        vp = vp.readlines()
        vp = [float(x) for x in vp]
        # ipm_img = Image.open(self.data_path+'/'+fname+'ipm.jpg')
        # mask = sio.loadmat(self.data_path+'/'+fname+'mask.mat')
        # mask = mask['BW']
        # ind = os.path.splitext(fname)[0]
        # den = self.read_csv('den', ind)
        # depth = self.read_csv('depth', ind)
        # ann = self.read_csv('ann', ind)
        # bbox = self.read_csv('bbox', ind)
        return img, vp


def getData(cfg):
    mean_std = cfg.DATA.MEAN_STD
    # log_para = cfg.LOG_PARA
    # factor = cfg.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    # depth_transform = standard_transforms.Compose([
    #     standard_transforms.ToTensor(),
    # ])
    # gt_transform = standard_transforms.Compose([
    #     own_transforms.GTScaleDown(factor),
    #     own_transforms.LabelNormalize(log_para)
    # ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    if cfg.DATA.NAME =='SelfLane':
        trainset = SelfLane(cfg.DATA.TRAIN_LIST)
        valset = SelfLane(cfg.DATA.VAL_LIST)
        trainloader = DataLoader(trainset, 
                                num_workers=cfg.DATA.NUM_WORKS,
                                batch_size=cfg.DATA.TRAIN_IMGBS, 
                                shuffle=cfg.DATA.IMGSHUFFLE)
        valloader = DataLoader(valset, 
                            num_workers=cfg.DATA.NUM_WORKS,
                            batch_size=cfg.DATA.VAL_IMGBS, 
                            shuffle=cfg.DATA.IMGSHUFFLE)
        meanImg = sio.loadmat(cfg.DATA.MEAN_IMG_PATH)
        meanImg = meanImg['meanImg']
        return meanImg,trainloader, valloader
    if cfg.DATA.NAME == 'TuSimpleLane':
        trainset = TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.TRAIN_LIST,isTrain=True)
        valset =TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.VAL_LIST,isTrain=False)
        trainloader =DataLoader(trainset,batch_size=cfg.DATA.TRAIN_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        valloader =DataLoader(valset,batch_size=cfg.DATA.VAL_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        meanImg =np.load(cfg.DATA.MEAN_IMG_PATH)
        return meanImg,trainloader,valloader
    if cfg.DATA.NAME == 'TRANCOS':
        trainset = TRANCOS(cfg.DATAROOT+'train', 'train', main_transform=train_main_transform, img_transform=img_transform, gt_transform=None, depth_transform=None)
        valset =TRANCOS(cfg.DATAROOT+'/test', 'test', main_transform=None, img_transform=img_transform, gt_transform=None, depth_transform=None)
        trainloader =DataLoader(trainset,batch_size=cfg.DATA.TRAIN_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        valloader =DataLoader(valset,batch_size=cfg.DATA.VAL_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        # meanImg =np.load(cfg.DATA.MEAN_IMG_PATH)
        return trainloader,valloader,restore_transform
    if cfg.DATA.NAME == 'UADETRAC':
        trainset = UADETRAC(cfg.DATAROOT+'IMGS', 'train', main_transform=train_main_transform, img_transform=img_transform, gt_transform=None, depth_transform=None)
        valset =UADETRAC(cfg.DATAROOT+'IMGS', 'test', main_transform=None, img_transform=img_transform, gt_transform=None, depth_transform=None)
        trainloader =DataLoader(trainset,batch_size=cfg.DATA.TRAIN_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        valloader =DataLoader(valset,batch_size=cfg.DATA.VAL_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        # meanImg =np.load(cfg.DATA.MEAN_IMG_PATH)
        return trainloader,valloader,restore_transform

class TuSimpleLane(data.Dataset):
    def __init__(self, dataroot, ListPath, isTrain=True,im_tf=None, gt_tf=None):
        if isTrain:
            self.root = osp.join(dataroot, 'train')
        else:
            self.root = osp.join(dataroot, 'test')
        self.root = osp.join(self.root, 'DRL', 'resize')
        with open(ListPath, 'r') as f:
            self.pathList= json.load(f)
        self.im_tf = im_tf
        self.gt_tf = gt_tf

    def __getitem__(self, index):
        # img
        temp = osp.join(self.root, self.pathList[index] + '.png')
        img = np.array(Image.open(temp))
        temp = osp.join(self.root, self.pathList[index] + '.json')
        with open(temp, 'r') as f:
            data = json.load(f)
        img = np.array(img)
        img = img.astype(np.float32)
        cla = np.array(data['class'])
        gt = np.array(data['gt'])
        return cla, img, gt

    def __len__(self):
        return len(self.pathList)


class SelfLane(data.Dataset):
    def __init__(self, pathList, im_tf=None, gt_tf=None):
        self.pathList = pathList
        self.im_tf = im_tf
        self.gt_tf = gt_tf

    def __getitem__(self, index):
        temp = sio.loadmat(self.pathList[index])
        img = temp['img']
        img = np.array(Image.fromarray(img).resize((100,100)))
        img =img.astype(np.float32)
        # fea = temp['fea']
        cl = np.array(int(temp['class_name'][0]))
        gt = np.array(temp['mark'][0])
        return cl, img, gt

    def __len__(self):
        return len(self.pathList)


class bufferLoader(data.Dataset):
    def __init__(self, buffer, tf=None):
        self.buffer = buffer
        self.tf = tf

    def __getitem__(self, index):
        fea, state, Q = self.buffer[index]
        fea = np.array(fea).astype(np.float32)
        state = np.array(state).astype(np.float32)
        Q = np.array(Q).astype(np.float32)
        if self.tf is not None:
            fea = self.tf(fea)
        return fea, state, Q

    def __len__(self):
        return len(self.buffer)
