# coding=utf-8
# ----tuzixini----
# WIN10 Python3.6.6
# tools/utils.py
'''
存放一些通用的工具
'''
import os
import cv2
import pdb
import time
import copy
import torch
import shutil
import random
import collections
import numpy as np
import numbers
# import random
# import numpy as np
from torchvision.transforms.functional import to_tensor, rotate
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch.nn as nn
# import torch
def initialize_weights(models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):    
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print( m )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.clean()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clean(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


def copy_cur_env(work_dir, dst_dir, exception='exp'):
    # 复制本次运行的工作环境,排除部分文件夹
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for filename in os.listdir(work_dir):
        file = os.path.join(work_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        if os.path.isdir(file) and exception not in filename:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file, dst_file)


class ExBuffer(object):
    def __init__(self, capacity, flashRat=1):
        self.cap = capacity
        self.replaceInd = 0
        self.replaceMax = int(capacity*flashRat)
        self.buffer = collections.deque(maxlen=capacity)

    def append(self, exp):
        self.buffer.append(exp)
        self.replaceInd += 1

    @property
    def isFull(self):
        if len(self.buffer) < self.cap:
            return False
        else:
            return True

    @property
    def ready2train(self):
        if self.isFull:
            if self.replaceInd < self.replaceMax:
                return False
            else:
                self.replaceInd = 0
                return True
        else:
            return False
    

    def samlpe(self, BS):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

    def clean(self):
        self.buffer.clear()
        self.replaceInd = 0


def visOneLane(img, gt, initY, initX, xpoints):
    # xpoints{'5':[2,3,4],'4':[1,1,1]}
    margin = 150
    temp = np.ones((margin*2+540, margin*2+960, 3)) * 255
    temp[margin:margin+540, margin:margin+960,:] = img
    initY = np.array(initY)
    initY = initY + margin
    initX = np.array(initX)
    initX = initX + margin
    gt = gt+margin
    xpoints=[x+margin for x in xpoints]
    # for i in xpoints.keys():
    #     xpoints[i] = np.array(xpoints[i]) + margin
    finalImg = []
    # 画gt
    # for i in range(5):
    # if gt>0:
    x = initX
    y = gt
    pt = (int(x),int(y))
    cv2.circle(temp, pt, 5, (0, 0, 255), 2)
    x1 = initX
    y_init = initY
    pt_init = (int(x1),int(y_init))
    cv2.circle(temp, pt_init, 6, (0, 255, 0), 2)
    pt_last = int(y_init)
    cir_ind = 0
    init_width = 15
    for p in xpoints:
        x2 = initX
        y_pred = p
        if p!=pt_last:
            if init_width!=1:
                init_width-=1
        pt_pred = (int(x2),int(y_pred))
        cv2.circle(temp, pt_pred, init_width, (255, 0, 0), 2)
        cir_ind+=1
        # cv2.line(temp, pt_last, (int(x2+1.5*(p+1)),int(y_pred)), (255, 0, 0), 2)
        pt_last = (int(x2+1.5*(p+2)),int(y_pred))
    # 五行
    oneLine = []
    oneLine.append(temp)
    finalImg.append(oneLine)
    return finalImg

def catFinalImg(finalImg):
    maxlen = 0
    for line in finalImg:
        if len(line) > maxlen:
            maxlen = len(line)
    x = finalImg[0][0].shape[0]
    w = finalImg[0][0].shape[1]
    h = x*maxlen
    img = np.ones((h, w, 3)) * 255
    for i in range(len(finalImg)):
        for j in range(len(finalImg[i])):
            img[x * i:x * (i + 1), w* j:w * (j + 1),:] = finalImg[i][j]
    return img


# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, vp, depth=None, ann=None, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, vp = t(img, vp)
            return img, vp
        else:
            for t in self.transforms:
                img, mask, depth, ann, bbx = t(img, mask, depth, ann, bbx)
            return img, mask, depth, ann, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, vp, ipm=None, mask=None, depth=None, ann=None, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                w, h = img.size
                return img.transpose(Image.FLIP_LEFT_RIGHT), [w-vp[0], vp[1]]
            w, h = img.size
            xmin = w - bbx[:, 2]
            xmax = w - bbx[:, 0]
            bbx[:, 0] = xmin
            bbx[:, 2] = xmax
            ann[:, 0] = w - ann[:, 0]
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT), ann, bbx
        if bbx is None:
            return img, vp
        return img, vp, depth, ann, bbx

class ToTensor(object):
    def __call__(self, img, mask):
        return to_tensor(img), to_tensor(mask)

class RandomRotation(object):
    def __call__(self, img, mask):
        assert torch.is_tensor(img)
        if random.random() < 0.5:
            if len(mask.shape)==2:
                mask = mask.unsqueeze(0)
            angle = random.random() * 60 - 30
            img = rotate(img, angle)
            mask = rotate(mask, angle)
            return img, mask.squeeze()
        else:
            return img, mask

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return  mask.resize((self.size[1]/cfg.TRAIN.DOWNRATE, self.size[0]/cfg.TRAIN.DOWNRATE), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print( img.size )
            print( mask.size )          
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        # a = torch.unique(tensor)
        # t = np.array(tensor, dtype=np.float)
        tensor = tensor*self.para
        # b = torch.unique(tensor)
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w//self.factor, h//self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img
