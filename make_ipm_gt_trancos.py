import numpy as np
import torch
from numba import cuda, jit
import os
import math
from PIL import Image, ImageOps
from tqdm import tqdm
import time
from visdom import Visdom
# vis = Visdom(port=7766)
@jit(nopython=True)
def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
    LBPoint = AreaPoint[0] #AreaPoint为按顺时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    LTPoint = AreaPoint[1]
    RTPoint = AreaPoint[2]
    RBPoint = AreaPoint[3]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    #print(a,b,c,d)
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False
@jit(nopython=True)  
def theta_compute(V_y, alpha, N):
    theta = np.arctan(np.tan(alpha)*(1-2*V_y/N))
    return theta
@jit(nopython=True)
def compute_xy(Height, theta, alpha, beta, i, j, W, H):
    #沿对角线对称
    u, v = j, i
    w_y = abs( Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*u/(W-1)))) )
    w_x = np.sqrt(Height*Height+w_y*w_y)*np.tan(beta)*(2*v/(H-1)-1)/np.sqrt(1+pow(np.tan(alpha)*(1-2*u/(W-1)), 2))

    # w_x = np.ceil(w_x)
    # w_y = np.ceil(w_y)
    return w_x, w_y

@jit(nopython=True)
def find_max(Height, theta, alpha, H, W):
    max_y, max_j = 0, 0
    for x in range(H):
        if x%100 == 0:
            y = abs( Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*x/(W-1)))) )
            if y>max_y:
                max_y, max_j = y, x
    return max_y, max_j

@jit(nopython=True)
def IPM( input, point, out_size,boxes):
    #基本参数和RGB数值
    RGB0, RGB1, RGB2 = input[0,:,:],input[1,:,:],input[2,:,:]
    H, W= RGB0.shape[0], RGB0.shape[1]
    alpha, beta, Height = math.radians(30), math.radians(45), 100
    h_out, w_out = out_size[0], out_size[1]

    #计算垂直偏转角和必要的框架数据
    theta = theta_compute(point, alpha, H)
    # gamma = self.gamma_compute(point[0], beta, W)
    x3, y3 = compute_xy(Height, theta, alpha, beta, 0, H, W, H)
    x4, y4 = compute_xy(Height, theta, alpha, beta, W, H, W, H)
    if point>0:
        # w_y = abs(Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*point/(W-1)))))
        # max_j = H-point
        # max_Y = abs( Height/np.tan(theta-np.arctan(np.tan(alpha)*(1-2*max_j/(W-1)))) )
        # min_X = 0
        # max_j = 0.5*(W-1)*(1-np.tan(theta-(np.pi/2-np.arctan(max_Y/Height)))/np.tan(alpha))
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
    h_scale_factor = res_h/h_out;
    w_scale_factor = res_w/w_out;  
    BOUNDX, BOUNDY = [(x-min_X)/w_scale_factor for x in BOUNDX], [(-y+max_Y)/h_scale_factor for y in BOUNDY]    
    res = np.zeros((3, h_out, w_out))
  
    ####################IPM#####################
    # for i in range(W):
    #     for j in range(math.ceil(max_j), H):
    #         x, y = compute_xy(Height, theta, alpha, beta, i, j, W, H)
    #         x, y=(x-min_X)/w_scale_factor, (max_Y-y)/h_scale_factor
    #         value1 = RGB0[j][i]
    #         value2 = RGB1[j][i]
    #         value3 = RGB2[j][i]
    #         res[0][int(y)][int(x)],res[1][int(y)][int(x)],res[2][int(y)][int(x)]=value1,value2,value3


    for i in range(w_out):
        for j in range(h_out):
            if isInterArea([i,j], [[BOUNDX[0],BOUNDY[0]], [BOUNDX[1], BOUNDY[1]], [BOUNDX[2], BOUNDY[2]], [BOUNDX[3],BOUNDY[3]]]):
                org_h, org_w = max_Y-j*h_scale_factor, i*w_scale_factor+min_X
 
                pixel_u = 0.5*(W-1)*(1-np.tan(theta-(np.pi/2-np.arctan(org_h/Height)))/np.tan(alpha))
                pixel_v = 0.5*(H-1)*(1+org_w/(np.tan(beta)*np.sqrt(Height*Height+org_h*org_h))*np.sqrt(1+pow(np.tan(theta-(np.pi/2-np.arctan(org_h/Height))), 2)))
                res_u, res_v = pixel_v, pixel_u
                if 0<=res_u<=W-1 and 0<=res_v<=H-1:
                    X0, X1, Y0, Y1 = int(np.floor(res_u)), int(np.ceil(res_u)), int(np.floor(res_v)), int(np.ceil(res_v))
                    u = res_u-X0
                    v = res_v-Y0
                    value1 = v*(u*RGB0[Y1][X1]+(1-u)*RGB0[Y1][X0])+(1-v)*(u*RGB0[Y0][X1]+(1-u)*RGB0[Y0][X0])
                    value2 = v*(u*RGB1[Y1][X1]+(1-u)*RGB1[Y1][X0])+(1-v)*(u*RGB1[Y0][X1]+(1-u)*RGB1[Y0][X0])
                    value3 = v*(u*RGB2[Y1][X1]+(1-u)*RGB2[Y1][X0])+(1-v)*(u*RGB2[Y0][X1]+(1-u)*RGB2[Y0][X0])
                    res[0][j][i], res[1][j][i], res[2][j][i] = value1, value2, value3
    return res

def make_trancos_gt(data_path, out_dir, out_size):
    data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename)) and filename.split('.')[-1]=='jpg']
    count = 0
    for file in tqdm(data_files):
        img = np.array(Image.open(os.path.join(data_path, file))).transpose(2, 0, 1)
        file_name = file.split('.')[0]
        vp = open(os.path.join(data_path, file_name+'vl.txt'))
        vp = vp.readlines()
        vp = [float(x) for x in vp][1]
        start_time = time.time()
        ipm_img = np.uint8(IPM(img, vp, out_size))
        # a = ipm_img.transpose(1, 2, 0)
        # ipm_img.dtype=np.int8
        ipm_img = Image.fromarray(ipm_img.transpose(1, 2, 0),mode='RGB')
        # ipm_img.show()
        # vis.image(ipm_img)
        end_time = time.time()
        print("处理当前图像共耗费{:.2f}秒,是第{}张, ID为{}".format(end_time-start_time, count, file))
        count+=1
        if os.path.exists(out_dir):
            ipm_img.save(os.path.join(out_dir, file_name+'ipm.jpg'))
        else:
            os.mkdir(out_dir)
            ipm_img.save(os.path.join(out_dir, file_name+'ipm.jpg'))
    return


if __name__=='__main__':
    train_data_path = '/data/haojie/TRANCOS/train/'
    test_data_path = '/data/haojie/TRANCOS/test/'
    out_train_dir = '/data/haojie/TRANCOS/train/'
    out_test_dir = '/data/haojie/TRANCOS/test/'
    out_size = (640, 480)
    make_trancos_gt(train_data_path, out_train_dir, out_size)
    make_trancos_gt(test_data_path,  out_test_dir, out_size)
