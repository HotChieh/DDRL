from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
# from misc.layer import Conv2d, FC
from torchvision import models
from utils import *
import torchvision.ops
# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class VGG16_Unet(nn.Module):
    
    def __init__(self):
        super(VGG16_Unet, self).__init__()
        
        
        vggfeat = list(models.vgg16(pretrained = False).features)
        self.feature1 = nn.Sequential(*vggfeat[0:4])
        self.feature2 = nn.Sequential(*vggfeat[4:9])
        self.feature3 = nn.Sequential(*vggfeat[9:16])
        self.feature4 = nn.Sequential(*vggfeat[16:23])
        self.feature5 = nn.Sequential(*vggfeat[23:30])
        del vggfeat
        
        self.decode1 = nn.Sequential(Conv2d(128, 64, kernel_size = 3, padding = 1))
        self.decode2 = nn.Sequential(Conv2d(256, 128, kernel_size = 3, padding = 1),
                                      Conv2d(128, 64, kernel_size = 3, padding = 1))
        self.decode3 = nn.Sequential(Conv2d(512, 256, kernel_size = 3, padding = 1),
                                      Conv2d(256,128, kernel_size = 3, padding = 1))
        self.decode4 = nn.Sequential(Conv2d(1024, 512, kernel_size = 3, padding = 1),
                                      Conv2d(512, 256, kernel_size = 3, padding = 1))
                                
        self.prob_conv = nn.Sequential(Conv2d(64, 1, kernel_size = 3, padding = 1))  
        
        self.initialize_weights()
        mod = models.vgg16(pretrained = True)

        len1 = len(self.feature1.state_dict().items())
        len2 = len1 + len(self.feature2.state_dict().items())
        len3 = len2 + len(self.feature3.state_dict().items())
        len4 = len3 + len(self.feature4.state_dict().items())
                                
        for i in range(len(self.feature1.state_dict().items())):
            list(self.feature1.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        for i in range(len(self.feature2.state_dict().items())):
            list(self.feature2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len1][1].data[:]
        for i in range(len(self.feature3.state_dict().items())):
            list(self.feature3.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len2][1].data[:]
        for i in range(len(self.feature4.state_dict().items())):
            list(self.feature4.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len3][1].data[:]
        for i in range(len(self.feature5.state_dict().items())):
            list(self.feature5.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i+len4][1].data[:]
            
        del mod        

        
    def forward(self, im_data):
        _, _,H,W = im_data.shape
        feature1 = self.feature1(im_data) #1/1
        _, _,h1,w1 = feature1.shape#1/1
        feature2 = self.feature2(feature1)#1/2
        _, _,h2,w2 = feature2.shape
        feature3 = self.feature3(feature2)#1/4
        _, _,h3,w3 = feature3.shape
        feature4 = self.feature4(feature3)#1/8
        _, _,h4,w4 = feature4.shape
        feature5 = self.feature5(feature4)     #1/16   
        
        up_feature5 = nn.functional.interpolate(feature5, size = (h4, w4), mode = "bilinear", align_corners = True)
        cat_feature4 = torch.cat((feature4, up_feature5), 1)
        de_feature4 = self.decode4(cat_feature4)
        del feature5, up_feature5, feature4, cat_feature4
        
        up_feature4 = nn.functional.interpolate(de_feature4, size = (h3, w3), mode = "bilinear", align_corners = True)
        cat_feature3 = torch.cat((feature3, up_feature4), 1)
        de_feature3 = self.decode3(cat_feature3)
        del de_feature4, up_feature4, feature3, cat_feature3
        
        up_feature3 = nn.functional.interpolate(de_feature3, size = (h2, w2), mode = "bilinear", align_corners = True)
        cat_feature2 = torch.cat((feature2, up_feature3), 1)
        de_feature2 = self.decode2(cat_feature2)
        del de_feature3, up_feature3, feature2, cat_feature2
        
        up_feature2 = nn.functional.interpolate(de_feature2, size = (h1, w1), mode = "bilinear", align_corners = True)
        cat_feature1 = torch.cat((feature1, up_feature2), 1)
        de_feature1 = self.decode1(cat_feature1)        
        del de_feature2, up_feature2, feature1, cat_feature1
            
        prob_map = self.prob_conv(de_feature1)
        
        #prob_map = torch.clamp(prob_map,0,1)
        
        return prob_map

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()


        self.fuse1_r = nn.Sequential(DeformableConv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse2_r = nn.Sequential(DeformableConv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse3_r = nn.Sequential(DeformableConv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        # self.fuse4_r = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1),nn.ReLU(inplace=True))
        self.c64_1 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c64_2 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.c64_3 = nn.Sequential(DeformableConv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.self_multi_64 = MultiHeadAttention(query_dim=64, key_dim=64, num_units=64, num_heads=4)
        # self.ca_64 = ChannelAttention(64)
        self.c32_1 = nn.Sequential(DeformableConv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c32_2 = nn.Sequential(DeformableConv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.c32_3 = nn.Sequential(DeformableConv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.self_multi_32 = MultiHeadAttention(query_dim=32, key_dim=32, num_units=32, num_heads=4)
        self.cross_multi_64_32 = MultiHeadAttention(query_dim=64, key_dim=32, num_units=32, num_heads=4)
        # self.ca_32 = ChannelAttention(32)

        self.p1 = nn.Sequential(DeformableConv2d(32, 1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.p2 = nn.Sequential(DeformableConv2d(32, 1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.p3 = nn.Sequential(DeformableConv2d(32, 1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        
        self.relu = nn.ReLU(inplace=True)
        self.b1 = nn.Sequential(DeformableConv2d(32, 2, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(DeformableConv2d(32, 2, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(DeformableConv2d(32, 2, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        initialize_weights(self.modules())
        vgg = models.vgg16_bn(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features1 = nn.Sequential(*features[0:6])
        self.features2 = nn.Sequential(*features[6:13])
        self.features3 = nn.Sequential(*features[13:23])
        self.features4 = nn.Sequential(*features[23:33])
        self.features5 = nn.Sequential(*features[33:43])

    def forward(self, x):

        x_pre = self.features1(x)  # 64 
        x1 = self.features2(x_pre)   # 128 1/2
        x2 = self.features3(x1)   # 256  1/4
        x3 = self.features4(x2)  # 512  1/8
        x4 = self.features5(x3) # 512 1/16

        x1 = self.fuse1_r(x2)
        x2 = self.fuse2_r(x3)
        x3 = self.fuse3_r(x4)

        x1_64 = self.c64_1(x1)
        x1_64_ca = self.self_multi_64(x1_64, x1_64)  
        x1_64 = x1_64_ca      
        x2_64 = self.c64_2(x2)
        x2_64_ca = self.self_multi_64(x2_64, x2_64)
        x2_64 = x2_64_ca
        x3_64 = self.c64_3(x3)
        x3_64_ca = self.self_multi_64(x3_64, x3_64)
        x3_64 = x3_64_ca

        x1_32 = self.c32_1(x1_64)
        x1_32_ca = self.self_multi_32(x1_32, x1_32) 
        x1_32_ca_cross = self.cross_multi_64_32(x1_64, x1_32) 
        x1_32 = x1_32_ca+x1_32_ca_cross      
        x2_32 = self.c32_2(x2_64)
        x2_32_ca = self.self_multi_32(x2_32, x2_32)
        x2_32_ca_cross = self.cross_multi_64_32(x2_64, x2_32) 
        x2_32 = x2_32_ca+x2_32_ca_cross
        x3_32 = self.c32_3(x3_64)
        x3_32_ca = self.self_multi_32(x3_32, x3_32)
        x3_32_ca_cross = self.cross_multi_64_32(x3_64, x3_32) 
        x3_32 = x3_32_ca+x3_32_ca_cross

        b_1 = self.b1(x1_32)
        b_2 = self.b2(x2_32)
        b_3 = self.b3(x3_32)

        B2 = self.up_sample(b_3, out_target=b_2)+b_2
        B1 = self.up_sample(B2, out_target=b_1)+b_1
        b = self.up_sample(b_3, out_target=b_1)+self.up_sample(B2, out_target=b_1)+B1  
        bbox_out = self.up_sample(b, out_target=x)


        p3 = self.p3(x3_32)
        p2 = self.p2(x2_32)
        p1 = self.p1(x1_32)

        p2 = self.up_sample(p3, out_target=p2)+p2
        p1 = self.up_sample(p2, out_target=p1)+p1

        p = self.up_sample(p3, out_target=p1)+self.up_sample(p2, out_target=p1)+p1  

        dm = self.up_sample(p, out_target=x)


        return dm, bbox_out

    def up_sample(self, x, out_target):
        _, _, w, h = out_target.size()
        x = F.interpolate(x, size=(w, h), mode='bilinear')
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        # modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        # self.regular_conv.weight = self.regular_conv.weight.half() if x.dtype == torch.float16 else \
        #     self.regular_conv.weight
        x = torchvision.ops.deform_conv2d(input=x.float(),
                                          offset=offset.float(),
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=(self.padding, self.padding),
                                          # mask=modulator,
                                          stride=self.stride,
                                          )
        return x
    

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim] 
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
 
    def __init__(self, query_dim, key_dim, num_units, num_heads):
 
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
 
        self.W_query = nn.Conv2d(query_dim, num_units,kernel_size=1, stride=1)
        self.W_key = nn.Conv2d(key_dim, num_units,kernel_size=1, stride=1)
        self.W_value = nn.Conv2d(key_dim, num_units,kernel_size=1, stride=1)
 
    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
        b, c, h, w = values.shape
        querys = querys.view(querys.shape[0], querys.shape[1], -1)
        keys = keys.view(keys.shape[0], keys.shape[1], -1)
        values = values.view(values.shape[0], values.shape[1], -1)


        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=1), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=1), dim=0)  # [h, N, T_k, num_units/h]
 
        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
 
        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads,1,querys.shape[2],1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)
 
        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=2).squeeze(0)  # [N, T_q, num_units]
        out = out.view(b, c, h, w)
        return out