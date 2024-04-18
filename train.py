# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# train.py
import os
import pdb
import torch
import scipy
import random
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image

from config import cfg
import utils
import datasets
import model
import reward
from utils import Timer
from visdom import Visdom

import torch.multiprocessing as mp
from torch.multiprocessing import Barrier

# vis = Visdom(env='main', port=1183)
# vis.line([[0.]], [0], win='q-value-loss', opts=dict(title='q-value-loss', legend=['train']))
# vis.line([[0.],[0.]], [0], win='train-val-sim', opts=dict(title='train-val-sim', legend=['train','val']))
# vis.line([[0.],[0.]], [0], win='train-val-ratio', opts=dict(title='train-val-ratio', legend=['train','val']))
# vis.line([[0.],[0.]], [0], win='train-val-dis', opts=dict(title='train-val-dis', legend=['train','val']))
class trainer(object):
    def __init__(self, cfg):
        self.epoch_iter = 0
        self.train_epoch = 0
        self.cfg = cfg
        os.makedirs(self.cfg.EXP.PATH, exist_ok=True)
        os.makedirs(self.cfg.EXP.PATH+'/valimg', exist_ok=True)
        # logger
        self.writer = SummaryWriter(self.cfg.EXP.PATH)
        # 计时器
        self.t = {'iter': Timer(), 'train': Timer(), 'val': Timer()}
        # 保存实验环境 # TODO: 启用
        temp = os.path.join(self.cfg.EXP.PATH, 'code')
        utils.copy_cur_env('./', temp, exception='exp')
        # 读取数据集
        self.trainloader, self.valloader, self.restore_transform = datasets.getData(self.cfg)
        # 定义网络
        self.net = model.getModel(cfg)
        # 损失函数
        self.criterion = torch.nn.MSELoss()
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # 初始化一些变量
        self.beginEpoch = 1
        self.batch = 1
        self.bestacc = 0
        self.bestdis = 1000
        # 载入预训练模型
        if self.cfg.TRAIN.RESUME:
            print('Loading Model..........')
            saved_state = torch.load(self.cfg.TRAIN.RESUME_PATH)
            self.net.load_state_dict(saved_state['weights'])
            self.beginEpoch = saved_state['epoch']
            self.batch = saved_state['batch']
            self.bestacc = saved_state['bestacc']
        # GPU设定
        self.gpu = torch.cuda.is_available() and self.cfg.TRAIN.USE_GPU
        self.device = 'cuda' if self.gpu else 'cpu'
        if self.gpu:
            torch.cuda.set_device(self.cfg.TRAIN.GPU_ID[0])
            self.criterion.cuda()
            if len(self.cfg.TRAIN.GPU_ID) > 1:
                self.net = torch.nn.DataParallel(
                    self.net, device_ids=self.cfg.TRAIN.GPU_ID)
            self.net = self.net.cuda()
        else:
            self.net = self.net.cpu()
            self.criterion.cpu()
        self.q_iter = 0
    def train(self):
        for self.epoch in range(self.beginEpoch, self.cfg.TRAIN.MAX_EPOCH):
            # 训练一个Epoch
            # acc = self.val()
            # acc, dis_sim = self.val()
            self.t['train'].tic()
            train_sim, train_ratio, train_dis = self.trainEpoch()
            temp = self.t['train'].toc(average=False)
            print('Train time of Epoch {} is : {:.2f}s'.format(self.epoch, temp))
            # 在验证集上测试
            self.t['val'].tic()
            acc, test_ratio, test_sim, test_dis = self.val(self.epoch)
            # vis.line([[train_sim], [test_sim]], [self.epoch], win='train-val-sim', update='append')
            # vis.line([[train_ratio], [test_ratio]], [self.epoch], win='train-val-ratio', update='append')
            # vis.line([[train_dis], [test_dis]], [self.epoch], win='train-val-dis', update='append')
            
            # vis.line([[train_dis], [dis_avg]], [self.epoch], win='train-val-dis', update='append')
            temp = self.t['val'].toc(average=False)
            print('Val time of/after Epoch {} is : {:.2f}s'.format(self.epoch, temp))
            self.writer.add_scalar('ValHitRate_PerEpoch', acc, self.epoch)
            # 保存模型
            if test_dis < self.bestdis:  
                self.bestdis = test_dis
                temp = "Dis{:.5f}".format(test_dis)
                self.save(temp)
            if acc>self.bestacc:
                self.bestacc = acc
            print('Acc for Epoch {} is : {:.4f}'.format(self.epoch, acc))
            print('BestDis is:{:.4f}'.format(self.bestdis))

    def trainEpoch(self):
        #建立一个参数池，容量为cfg.BUFFER_CAP，此处为20480*3
        self.buffer = utils.ExBuffer(self.cfg.BUFFER_CAP)
        #初始化参数池
        self.buffer.clean()
        print('Build Buffer.........')
        train_ratio = 0.0
        train_dis = 0.0
        train_sim=0.0
        train_item_count = 0
        for batch_index, (imgs, gts) in tqdm(enumerate(self.trainloader)):
            # clas = clas.numpy()
            imgs = imgs.numpy()
            gts = gts[1].numpy()
            for j in range(len(imgs)):
                self.img = imgs[j]
                # self.cl = clas[j]
                self.gt = gts[j]
                # if self.cl == 1:
                    # self.initMarkX = [91.0, 71.0, 51.0, 31.0, 11.0]
                # else:
                    # self.initMarkX = [11.0, 31.0, 51.0, 71.0, 91.0]
                self.initMarkX = imgs.shape[2]*1/4
                train_cur_ration, train_per_sim, train_per_dis = self.updateBuffer()
                train_item_count+=1
                train_sim+=train_per_sim
                train_dis+=train_per_dis
                train_ratio+=train_cur_ration
                if self.trainFlag:
                    self.trainBuffer()
                    print('Build Buffer.........')
        train_sim/=train_item_count
        train_dis/=train_item_count
        train_ratio/=train_item_count
        print("训练集上平均比例为：{:.2f}, 平均相似度为：{:.2f}， 平均距离为：{:.2f}".format(train_ratio, train_sim, train_dis))
        # if self.trainFlag:
        self.trainBuffer()
        #     print('Build Buffer.........')
        return train_sim, train_ratio, train_dis

    def trainBuffer(self):
        print('Training..........')
        self.net.train()
        self.train_epoch = 0
        # tf = transforms.ToTensor()
        dataset = datasets.bufferLoader(self.buffer.buffer)
        loader = DataLoader(dataset, num_workers=self.cfg.DATA.NUM_WORKS, batch_size=self.cfg.DATA.BS, shuffle=self.cfg.DATA.SHUFFLE)
        for epoch in tqdm(range(self.cfg.TRAIN.INER_EPOCH)):
            vis_loss= 0.0
            iter_count = 0
            for fea, state, Q in tqdm(loader):
                # fea =torch.from_numpy(fea)
                fea, state, Q = fea.to(self.device), state.to(self.device), Q.to(self.device)
                self.optimizer.zero_grad()
                output = self.net(fea, state).to(self.device)
                loss = self.criterion(output, Q)
                vis_loss+=loss.item()
                iter_count+=1
                self.q_iter += 1
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('trian_loss', loss.item(), self.batch)
                self.batch += 1
            # vis.line([[vis_loss/iter_count]], [self.epoch_iter], win='q-value-loss', update='append')
            self.epoch_iter+=1
            self.train_epoch+=1
            # acc, test_ratio, test_sim, test_dis=self.val(self.epoch, self.train_epoch)
            # if test_dis < self.bestdis:  
            #     self.bestdis = test_dis
            #     temp = "Dis{:.5f}".format(test_dis)
            #     self.save(temp)
        self.save_train()

    def val(self, Buffer_epoch = None, Train_epoch = None):
        self.net.eval()
        hit_cnt = 0
        detect_hit_cnt = 0
        test_cnt = 0
        sup_cnt = 0
        steps_cnt = 0
        sim_avg = 0.0
        dis_avg = 0.0
        ratio_avg=0.0
        for valIndex, (img, gt) in tqdm(enumerate(self.valloader)):
            
            img = np.squeeze(img.numpy())
            # cl = np.squeeze(cl.numpy())
            Xinit = np.squeeze(gt[0].numpy())
            gt = np.squeeze(gt[1].numpy())
            # Xinit = np.squeeze(gt[0].numpy())
            img = img
            initMarkX = img.shape[1]*1/4
            # 循环处理五个landmark point
            xpoints = []


            cur_x = []
            step = 0
            allActList = np.zeros(self.cfg.MAX_STEP)
            status = 1
            gt_point = gt
            fea_t = np.array(img)
            # fea_t = np.transpose(fea_t, (2, 0, 1))
            # fea_t = fea_t.astype(np.float32)
            fea_t = fea_t.reshape((1,fea_t.shape[0],fea_t.shape[1],fea_t.shape[2]))
            fea_t = torch.from_numpy(fea_t).cuda()
            cur_point = initMarkX
            cur_x.append(cur_point)
            if self.cfg.HIS_NUM == 0:
                hist_vec = []
            else:
                hist_vec = np.zeros([self.cfg.ACT_NUM * self.cfg.HIS_NUM])
            state = reward.get_state(cur_point, hist_vec)
            while (status == 1) & (step < self.cfg.MAX_STEP):
                step += 1
                state= state.astype(np.float32).reshape((1,-1))
                state = torch.from_numpy(state).cuda()
                qval = np.squeeze(self.net(fea_t, state).detach().cpu().numpy())
                action = (np.argmax(qval)) + 1
                allActList[step - 1] = action
                if action != 3:
                    if action == 1:
                        cur_point -= 15
                        if cur_point==0:
                            cur_point = cur_point - 15
                    elif action == 2:
                        cur_point += 15
                        if cur_point==0:
                            cur_point = cur_point - 15
                    cur_x.append(cur_point)
                else:
                    status = 0
                if self.cfg.HIS_NUM != 0:
                    hist_vec = reward.update_history_vector(
                        hist_vec, action)
                state = reward.get_state(cur_point, hist_vec)
            steps_cnt += step
            finalPoint = cur_point
            dis_avg+=abs(finalPoint-gt)
            final_ratio, finalsim = reward.get_dst(fea_t.squeeze().detach().cpu().numpy(), cur_point)
            sim_avg+=finalsim
            ratio_avg+=final_ratio
            # det_dst = abs(initMarkX-gt_point)
            # if det_dst < self.cfg.DST_THR:
            #     detect_hit_cnt += 1
            test_cnt += 1
            # if finalsim >= self.cfg.DST_THR:
            if final_ratio ==1.0:
                hit_cnt += 1   
            # if finalsim <= det_dst:
            #     sup_cnt += 1
            xpoints = cur_x
            
            img = np.array(self.restore_transform(torch.from_numpy(img)))
            finImg = utils.visOneLane(img, gt, initMarkX, Xinit, xpoints)
            finImg = utils.catFinalImg(finImg)
            epoch = str(Buffer_epoch)+'_'+str(Train_epoch)
            if not os.path.exists(osp.join(self.cfg.EXP.PATH,'valimg',str(epoch))):
                os.mkdir(osp.join(self.cfg.EXP.PATH,'valimg',str(epoch)))
            tempPath = osp.join(self.cfg.EXP.PATH,'valimg',str(epoch), 'val'+str(test_cnt)+'.png')
            Image.fromarray(finImg.astype('uint8')).save(tempPath)
            finImg=np.transpose(finImg, (2,0,1))
            self.writer.add_image('Val_Vis',finImg,self.epoch)
            self.writer.add_scalar('Val_RL_HR', float(hit_cnt) / test_cnt, self.epoch)
            self.writer.add_scalar('Val_Hit_Cnt',hit_cnt,self.epoch)
            self.writer.add_scalar('Val_Det_HR', float(detect_hit_cnt) / test_cnt, self.epoch)
            self.writer.add_scalar('Val_Det_Hit_Cnt',detect_hit_cnt,self.epoch)
            # self.writer.add_scalar('Val_RLsupDet_HR', float(sup_cnt)/test_cnt, self.epoch)
            self.writer.add_scalar('Val_Average_Step', float(steps_cnt) / ((valIndex + 1) * 5), self.epoch)
        sim_avg/=test_cnt
        dis_avg/=test_cnt
        ratio_avg/=test_cnt
        print("测试集上平均比例为：{:.2f}， 平均相似度为：{:.2f}, 平均距离为：{:.2f}，共有{}张达到了阈值要求".format(ratio_avg, sim_avg, dis_avg, hit_cnt))    
        return float(hit_cnt) / test_cnt, ratio_avg, sim_avg, dis_avg

    def updateBuffer(self):
        #更新参数池，此时不属于训练，所以trainflag是False
        self.trainFlag = False
        #(具名元组)；Point = namedtuple("Point", ['x', 'y'])；p1 = Point(2, 3)；p1.x  
        buf = collections.namedtuple('buf', field_names=['fea', 'state', 'Q'])
        # generateExpReplay
        gt_point = self.gt
        status = 1 #表示智能体存活
        step = 0
        cur_point = self.initMarkX
        landmark_fea = np.array(self.img)#图像
        if self.cfg.HIS_NUM == 0:
            hist_vec = []
        else:
            hist_vec = np.zeros([self.cfg.HIS_NUM*self.cfg.ACT_NUM])#历史数*动作数 8*3
        state = reward.get_state(cur_point, hist_vec)#初始位置和历史向量堆叠 尺寸25
        cur_dets_ratio, cur_sim = reward.get_dst(landmark_fea, cur_point)#计算当前位置的相似性
        last_sim = cur_sim
        last_dets_num = cur_dets_ratio
        
        while (status == 1) & (step < self.cfg.MAX_STEP):
            rew = []
            qval = np.array(self.predict(landmark_fea, state))
            step += 1
            # 挑选action 计算reward
            # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
            # if cur_sim > self.cfg.DST_THR and cur_dets_ratio>0.9:
            if cur_dets_ratio==1.0 and cur_sim>cfg.DST_THR:
                action = 3
            # epsilon-greedy policy
            elif random.random() < self.cfg.EPSILON:
                action = np.random.randint(1, 4)
            else:
                action = (np.argmax(qval)) + 1
            # terminal action
            if action == 3:
                rew = reward.get_reward_trigger(cur_dets_ratio)
            # move action,performe the crop of the corresponding subregion
            # elif action == 1:
            #     cur_point = -20
            #     cur_dst = reward.get_dst(gt_point, cur_point)
            #     rew = reward.getRewRm(cur_dst)
            #     last_dst = cur_dst
            #     last_point = cur_point
            elif action == 1:  # to up
                cur_point = cur_point - 15
                if cur_point==0:
                    cur_point = cur_point - 15
                cur_dets_ratio, cur_sim = reward.get_dst(landmark_fea, cur_point)
                rew = reward.getRewMov0427(cur_sim, cur_dets_ratio, last_sim, last_dets_num)
                last_sim = cur_sim
                last_dets_num = cur_sim
                last_point = cur_point
            elif action == 2:  # to down
                cur_point = cur_point +15
                if cur_point==0:
                    cur_point = cur_point - 15
                cur_dets_ratio, cur_sim = reward.get_dst(landmark_fea, cur_point)
                rew = reward.getRewMov0427(cur_sim, cur_dets_ratio, last_sim, last_dets_num)
                last_sim = cur_sim
                last_dets_num = cur_sim
                last_point = cur_point
            if self.cfg.HIS_NUM != 0:
                hist_vec = reward.update_history_vector(hist_vec, action)
            new_state = reward.get_state(cur_point, hist_vec)
            # 计算 用来训练的Q值
            if action == 3:
                temp = rew
            else:
                temp = np.array(self.predict(landmark_fea, new_state))
                temp = np.argmax(temp)
                temp = rew + self.cfg.GAMMA * temp
            qval[action-1] = temp
            # 将数据存入buffer
            if self.buffer.ready2train:
                self.trainFlag = True
                break
            else:
                temp = buf(landmark_fea, state, qval)
                self.buffer.append(temp)
            if action == 3:
                status = 0
            state = new_state
        cur_ratio, trin_per_sim = reward.get_dst(landmark_fea, cur_point)
        train_per_dis = abs(cur_point-self.gt)
        return cur_ratio, trin_per_sim, train_per_dis


    def predict(self, fea, sta):
        self.net.eval()
        # fea = np.transpose(fea, (2, 0, 1))
        # fea = fea.astype(np.float32)
        fea = fea.reshape((1,fea.shape[0],fea.shape[1],fea.shape[2]))
        fea = torch.from_numpy(fea).cuda()
        sta = sta.astype(np.float32)
        sta = sta.reshape((1,-1))
        sta = torch.from_numpy(sta).cuda()

        x = self.net(fea, sta)
        return np.squeeze(x.data.detach().cpu().numpy())

    def save_train(self):
        fname = 'latest_weights.pth'
        path = osp.join(self.cfg.EXP.PATH, fname)
        if (not self.cfg.TRAIN.USE_GPU) or (len(self.cfg.TRAIN.GPU_ID) == 1):
            to_saved_weight = self.net.state_dict()
        else:
            to_saved_weight = self.net.module.state_dict()
        toSave = {
            'weights': to_saved_weight,
            'epoch': self.epoch,
            'batch': self.batch,
            'bestacc': self.bestacc
        }
        torch.save(toSave, path)
        print('Train Model Saved!')

    def save(self,temp):
        temp = 'EP_'+str(self.epoch)+'_'+temp+'.pth'
        path = osp.join(self.cfg.EXP.PATH, temp)
        if (not self.cfg.TRAIN.USE_GPU) or (len(self.cfg.TRAIN.GPU_ID) == 1):
            to_saved_weight = self.net.state_dict()
        else:
            to_saved_weight = self.net.module.state_dict()
        toSave = {
            'weights': to_saved_weight,
            'epoch': self.epoch,
            'batch': self.batch,
            'bestacc': self.bestacc
        }
        torch.save(toSave, path)
        print('Model Saved!')


if __name__ == "__main__":
    utils.setup_seed(cfg.SEED)
    MyTrainer = trainer(cfg)
    MyTrainer.train()
