import math
import torch
import numpy as np
import torch.nn as nn
from VGG import VGG
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
from visdom import Visdom
from make_ipm_gt_trancos import IPM
from Res50_det import Res50
from numba import cuda, jit
import time



@jit(nopython=True)  
def box_ipm_affine(boxes, H, W, theta, Height, alpha, beta, vp_line):
    #将box的两个坐标进行IPM转换
    box_ipm_length = []
    box_ipm = []
    box_coor = []
    x3, y3 = compute_xy(Height, theta, alpha, beta, 0, H, W, H)
    x4, y4 = compute_xy(Height, theta, alpha, beta, W, H, W, H)
    if vp_line>0:
        w_y = abs(Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*vp_line/(W-1)))))
        min_X, max_Y = 0, 2*H
        max_j = 0.5*(W-1)*(1-np.tan(theta-(np.pi/2-np.arctan(max_Y/Height)))/np.tan(alpha))
        max_x1 = np.sqrt(Height*Height+max_Y*max_Y)*np.tan(beta)*(2*0/(H-1)-1)/np.sqrt(1+pow(np.tan(alpha)*(1-2*max_j/(W-1)), 2))
        max_x2 = np.sqrt(Height*Height+max_Y*max_Y)*np.tan(beta)*(2*W/(H-1)-1)/np.sqrt(1+pow(np.tan(alpha)*(1-2*max_j/(W-1)), 2))
        res_w = abs(np.ceil(max_x2-max_x1))
        res_h = abs(np.ceil(max_Y-y4))
        min_X= max_x1
        BOUNDX, BOUNDY = [max_x1, x3, x4, max_x2], [max_Y, y3, y4, max_Y]
    else:
        x1, y1 = compute_xy(Height, theta, alpha, beta, 0, 0, W, H)
        x2, y2 = compute_xy(Height, theta, alpha, beta, W, 0, W, H)
        res_w = abs(x2-x1)
        res_h = abs(y2-y4)
        min_X,max_Y= x1, y2
        BOUNDX, BOUNDY = [x1, x3, x4, x2], [y1, y3, y4, y2]
    
    h_scale_factor = res_h/H;
    w_scale_factor = res_w/W;  
    max_j = 0.5*(W-1)*(1-np.tan(theta-(np.pi/2-np.arctan(max_Y/Height)))/np.tan(alpha))
    for box in boxes:
        [x1, y1, x2, y2] = box
        # min_X, max_Y = 0, 2*H
        
        if y1>max_j:
            x1_t, y1_t = coor_trans(x1, y1, Height, theta, alpha, beta,W, H, min_X, max_Y, h_scale_factor, w_scale_factor, max_j)
            x2_t, y2_t = coor_trans(x2, y2, Height, theta, alpha, beta,W, H, min_X, max_Y, h_scale_factor, w_scale_factor, max_j)
            x3_t, y3_t = coor_trans(x2, y1, Height, theta, alpha, beta,W, H, min_X, max_Y, h_scale_factor, w_scale_factor, max_j)
            x4_t, y4_t = coor_trans(x1, y2, Height, theta, alpha, beta,W, H, min_X, max_Y, h_scale_factor, w_scale_factor, max_j)


            l2r_length = np.sqrt(pow(x2_t-x1_t, 2)+pow(y2_t-y1_t, 2))
            r2l_length = np.sqrt(pow(x2_t-x1_t, 2)+pow(y2_t-y1_t, 2))

            ipm_x1, ipm_x2 = min([x1_t, x2_t, x3_t, x4_t]), max([x1_t, x2_t, x3_t, x4_t])
            ipm_y1, ipm_y2 = min([y1_t, y2_t, y3_t, y4_t]), max([y1_t, y2_t, y3_t, y4_t])
            
            # box_ipm_w, box_ipm_h = ipm_x2-ipm_x1, ipm_y2-ipm_y1
            # box_ipm_length.append(np.sqrt(pow(box_ipm_w, 2)+pow(box_ipm_h, 2)))
            box_ipm_length.append(min([l2r_length, r2l_length]))
            ipm_box=[ipm_x1, ipm_y1, ipm_x2, ipm_y2]
            box_ipm.append(ipm_box)
            box_coor.append([x1_t, y1_t, x2_t, y2_t,x3_t, y3_t,x4_t, y4_t])
    return box_ipm_length, box_ipm, box_coor
@jit(nopython=True)  
def coor_trans(x, y, Height, theta, alpha, beta,W, H, min_X, max_Y, h_scale_factor, w_scale_factor, max_j):

    # org_h, org_w = max_Y-y*h_scale_factor, x*w_scale_factor+min_X
    # pixel_u = 0.5*(self.W-1)*(1-np.tan(self.theta-(np.pi/2-np.arctan(org_h/self.Height)))/np.tan(self.alpha))
    # pixel_v = 0.5*(self.H-1)*(1+org_w/(np.tan(self.beta)*np.sqrt(self.Height*self.Height+org_h*org_h))*np.sqrt(1+pow(np.tan(self.theta-(np.pi/2-np.arctan(org_h/self.Height))), 2)))
    # res_u, res_v = pixel_v, pixel_u

    x_t, y_t = compute_xy(Height, theta, alpha, beta, x, y, W, H)
    
    x_t, y_t = (x_t-min_X)/w_scale_factor, (max_Y-y_t)/h_scale_factor

    return x_t, y_t
@jit(nopython=True)  
def compute_xy(Height, theta, alpha, beta, i, j, W, H):
    #沿对角线对称
    u, v = j, i
    w_y = abs( Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*u/(W-1)))) )
    w_x = np.sqrt(Height*Height+w_y*w_y)*np.tan(beta)*(2*v/(H-1)-1)/np.sqrt(1+pow(np.tan(alpha)*(1-2*u/(W-1)), 2))

    # w_x = np.ceil(w_x)
    # w_y = np.ceil(w_y)
    return w_x, w_y

class similarity(object):
    def __init__(self):
        self.det_net = VGG(pretrained=False)
        # self.det_net = Res50(pretrained=False)
        
        # self.vp_line = point
        # self.state_dict = '/data/haojie/UADETRAC_train/exp/04-10_16-43_UADETRAC_Res50_0.0001/latest_state.pth'
        self.state_dict = './VGG.pth'
        self.restore_transform = standard_transforms.Compose([
        DeNormalize(*([0.3908707, 0.3613535, 0.36716083], [0.22240958, 0.21731742, 0.21530356])),
        # standard_transforms.ToPILImage()
    ])
        self.alpha, self.beta, self.Height = math.radians(30), math.radians(45), 100
        # self.theta = np.arctan(np.tan(self.alpha)*(1-2*self.vp_line/self.input.shape[2]))
        # self.H, self.W  = self.input.shape[1], self.input.shape[2]
        self.visual =True
        self.load_model(multi_GPU=True)
        torch.cuda.set_device(2)
        self.det_net.cuda()
        self.det_net.eval()
        
    def dets_sim(self, img, point):
        self.input = img
        self.H, self.W  = self.input.shape[1], self.input.shape[2]
        self.vp_line = point
        self.theta = np.arctan(np.tan(self.alpha)*(1-2*self.vp_line/self.H))
        # self.load_model(multi_GPU=True)
        with torch.no_grad():
            torch.cuda.set_device(2)
            self.input = Variable(self.input.unsqueeze(0)).cuda()

            heatmap, wh = self.det_net(self.input)
        
        #detection
        dets = self.wh2boxes(wh, heatmap/100)

        box_ipm_length, box_ipm, box_coor = box_ipm_affine(dets[0], self.H, self.W, self.theta, self.Height, self.alpha, self.beta, self.vp_line)
        box_ipm_length = torch.tensor(box_ipm_length)

        #可视化box
        if self.visual:
            self.vis(heatmap, dets, box_ipm_length, box_ipm, box_coor)

        gt = torch.ones_like(box_ipm_length)
        dets_sim = F.cosine_similarity(gt, box_ipm_length, dim = 0)
        # dets_sim = self.compute_dis(box_ipm_length)

        return  len(box_ipm_length)/len(dets[0]) if dets[0].any() else 0, dets_sim

    def compute_dis(self, dets):
        if len(dets)==0:
            return -1
        else:
            dets_t = dets.view(1, len(dets)).transpose(0, 1).expand(len(dets), len(dets))
            dets = dets.expand(len(dets), len(dets))
            dis = torch.pairwise_distance(dets, dets_t)


        return dis

    def vis(self, heatmap, dets, box_ipm_length, box_ipm, box_coor):
        ipm_img = IPM(np.array(self.restore_transform(self.input.detach().cpu().squeeze())), self.vp_line, (self.input.shape[2], self.input.shape[3]), dets[0])
        #行不通，先IPM再检测
        # de_det_input = Variable(img_transform(ipm_img.transpose(1, 2, 0))).to(self.input.device).float()
        # de_heatmap, de_wh = self.det_net(de_det_input.unsqueeze(0))
        # de_dets = self.wh2boxes(de_wh, de_heatmap/100)

        # #
        # fig1=plt.imshow(ipm_img.transpose(1,2,0))
        # for d in range(len(de_dets[0])):
        #     dd = de_dets[0][d]
        #     fig1.axes.add_patch(plt.Rectangle(xy=(dd[0], dd[1]), width=dd[2]-dd[0], height=dd[3]-dd[1], fill=False, edgecolor='blue', linewidth=1))
        # vis.matplot(plt, win='vis_ipm_de', opts=dict(title='vis_ipm_de'))
        # plt.clf()

        #原始输入加BOXES
        fig=plt.imshow(np.array(self.restore_transform(self.input[0].detach().cpu())).transpose(1,2,0))
        # vis.heatmap(heatmap[0].squeeze(), win='density_pred', opts=dict(title='density_pred'))
        for b in range(len(dets[0])):
            bb = dets[0][b]
            fig.axes.add_patch(plt.Rectangle(xy=(bb[0], bb[1]), width=bb[2]-bb[0], height=bb[3]-bb[1], linewidth=1, fill=False, edgecolor='blue'))
        # vis.matplot(plt, win='vis_bbox', opts=dict(title='vis_bbox'))
        plt.savefig('/data/haojie/TRANCOS_DQLL//635543_img00248.jpg',dpi=300)
        plt.clf()

        #IPM加BOX对角线
        fig0 = plt.imshow(ipm_img.transpose(1,2,0))
        for c in range(len(box_ipm)):
            box_c = box_coor[c]
            plt.plot([box_c[0],box_c[2], box_c[4],box_c[6]], [box_c[1],box_c[3], box_c[5],box_c[7]], 'r*')
            plt.plot([box_c[0],box_c[2]], [box_c[1],box_c[3]])
            plt.plot([box_c[4],box_c[6]], [box_c[5],box_c[7]])
            # cc = box_ipm[c]
            # fig0.axes.add_patch(plt.Rectangle(xy=(cc[0], cc[1]), width=cc[2]-cc[0], height=cc[3]-cc[1], fill=False, edgecolor='blue', linewidth=1))
        # vis.matplot(plt, win='vis_ipm', opts=dict(title='vis_ipm'))
        plt.clf()

    def wh2boxes(self, wh_pred, heatmap):
        hm = heatmap
        wh = wh_pred
        reg = None
        cat_spec_wh = False
        K = 50
        dets = self.ctdet_decode(hm, wh)
        b,n,box = dets.shape
        dets = dets.detach().cpu().numpy()
        top_pred = dets[:,:,4]>0.5
        det_res = []
        for i in range(len(top_pred)):
            top = top_pred[i]
            det = dets[i][top]
            if len(det) !=0:
                det = det[:,0:4]
            det_res.append(det)


        return det_res
    def ctdet_decode(self, heat, wh):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        # vis.heatmap(heat[0].squeeze(), win='heat_map_without_nms', opts=dict(title='heat_map_without_nms'))
        heat = self._nms(heat)
        # vis.heatmap(heat[0].squeeze(), win='heat_map_after_nms', opts=dict(title='heat_map_after_nms'))
        K = int(torch.ceil(torch.sum(heat)))
        scores, inds, clses, ys, xs = self._topk(heat, K)

        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
        wh = self._transpose_and_gather_feat(wh, inds)

        wh = wh.view(batch, K, 2)
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2, 
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def _nms(self, heat, kernel=25):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()
        
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds / width).int().float()
        topk_xs   = (topk_inds % width).int().float()
        
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat



    def load_model(self, multi_GPU):
        load_state_dict = torch.load(self.state_dict, map_location=torch.device('cuda:3'))
        new_state_dict = OrderedDict()
        for k,v in load_state_dict.items():  # single GPU load multi-GPU pre-trained model
            name = k.split(".")
            name.remove('CCN')
            if multi_GPU:
                name.remove('module')
            name = ".".join(name)
            new_state_dict[name] = v 
        self.det_net.load_state_dict(new_state_dict)






class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor





if __name__ == '__main__':
    # a = torch.FloatTensor([-4, -4, -4, -4])
    # b = torch.FloatTensor([1, 1, 1, 1])
    # c = F.cosine_similarity(a, b, dim=0)
    x1, y1, x2, y2 = 580, 380, 760, 540
    alpha, beta, Height = math.radians(30), math.radians(45), 100
    theta = np.arctan(np.tan(alpha)*(1-2*50/540))
    H, W= 540, 960
    x1_t, y1_t = compute_xy(Height, theta, alpha, beta, 0, H, W, H)
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*([0.3908707, 0.3613535, 0.36716083], [0.22240958, 0.21731742, 0.21530356]))
    ])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # img = torch.randn((1, 3, 480, 640))
    # point = list(range(135, -90, -15))
    point = [-55]
    # point = 120
    # img = Image.open('/data/haojie/UADETRAC/train/MVI_39931/img00004.jpg')
    img = Image.open('/data/haojie/UADETRAC/test/MVI_63563/img00248.jpg')
    # img = Image.open('./test.jpg')
    img = img_transform(img)
    img = Variable(img).to(device)
    sim = similarity()
    start_time = time.time()
    # vis.line([[0.],[0.]], [0], win='ratio_sim', opts=dict(title='ratio_sim', legend=['ratio','sim']))
    ind = 0
    test_time = 0
    for p in point:
        if p ==0:
            p=-15
        start_time = time.time()
        ratio, sim_num = sim.dets_sim(img, p)
        end_time = time.time()
        print(end_time-start_time)
        test_time+=(end_time-start_time)
        # vis.line([[ratio], [sim_num]], [ind], win='ratio_sim', update='append')
        ind+=1
    print('全部时间：', test_time)