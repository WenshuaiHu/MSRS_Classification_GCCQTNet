import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange

from torch.nn import init
from models.utils.mobile import Mobile, hswish, MobileDown
from models.utils.former import Former
from models.utils.bridge import Mobile2Former, Former2Mobile

from utils.modules.swg_transformer import SWG_Transformer
from models.corss_transformers import Attention_3_branches
        
class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim, device):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)    
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        
        return [x_out, z_out]

class BaseBlock_SAR(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, HSImodel, dim, device):
        super(BaseBlock_SAR, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        #'''
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)#'''

        self.former = HSImodel.former
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        
        return [x_out, z_out]
        
class GCCQTNet(nn.Module):
    def __init__(self, cfg, device, hsi_bands, sar_bands, patch_size, num_class):
        super(GCCQTNet, self).__init__()
        self.token = nn.Parameter(torch.empty(1, cfg['token'], cfg['embed']),
                                     requires_grad=True)  # Tokenization parameters  
        torch.nn.init.xavier_normal_(self.token)## 正态分布

        self.stem_hsi = nn.Sequential(
            nn.Conv2d(hsi_bands, cfg['stem'], kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )## 11x11x128
        self.stem_sar = nn.Sequential(
            nn.Conv2d(sar_bands, cfg['stem'], kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )## 11x11x4
        
        self.patch_size = patch_size
        self.num_class = num_class
        
        # body
        self.block1_0= nn.ModuleList()
        self.block2_0 = nn.ModuleList()
        self.block1 = nn.ModuleList()
        self.block2 = nn.ModuleList()
        for kwargs in cfg['body']:
            if kwargs['exp']==180:
                BaseBlock_1 = BaseBlock(**kwargs, dim=cfg['embed'], device=device)
                self.block1_0.append(BaseBlock_1)
                self.block2_0.append(BaseBlock_SAR(**kwargs, HSImodel = BaseBlock_1, dim=cfg['embed'], device=device))
            else:
                BaseBlock_1 = BaseBlock(**kwargs, dim=cfg['embed'], device=device)
                self.block1.append(BaseBlock_1)
                self.block2.append(BaseBlock_SAR(**kwargs, HSImodel = BaseBlock_1, dim=cfg['embed'], device=device))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        # self.conv = nn.Conv3d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn = nn.BatchNorm3d(exp)
        # self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.L = cfg['token']-1# num_tokens
        self.cT = cfg['embed'] #dim
        self.emb_dropout = 0.1

        # Tokenization for each channel 
        self.token_wA = nn.Parameter(torch.empty(1, self.L, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, self.cT, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, self.L+1, self.cT))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cT))
        self.dropout = nn.Dropout(self.emb_dropout)
        
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.cT*4), nn.Linear(self.cT*4, self.num_class))
        self.mlp_headz = nn.Sequential(nn.LayerNorm(self.cT), nn.Linear(self.cT, self.num_class))
        
        self.corss_fusion = Attention_3_branches(self.emb_dropout, cfg['embed'], ct_attn_heads = 8)
        self.init_params()        
    #'''
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)#'''

    def forward(self, x1, x2):

        b, _, _,_ = x1.shape  ## x1 HSI, x2 SAR b c s h w 
        z = self.token.repeat(b, 1, 1)### 
        x1 = self.stem_hsi(x1)## 11x11x244x8
        x2 = self.stem_sar(x2)## 11x11x4x8
        
        z1_ = []
        z2_ = []
        
        for m1, m2 in zip(self.block1_0, self.block2_0):
            x1, z1 = m1([x1, z])
            x2, z2 = m2([x2, z])
            z1_.append(z1[:, 0, :].view(b, -1))
            z2_.append(z2[:, 0, :].view(b, -1))
        for m1, m2 in zip(self.block1, self.block2):
            x1, z1 = m1([x1, z1])
            x2, z2 = m2([x2, z2])
            z1_.append(z1[:, 0, :].view(b, -1))
            z2_.append(z2[:, 0, :].view(b, -1))
            
        x1 = rearrange(x1,'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        #print(111111111, x1.shape, x2.shape)## 100 64 (13 13)
        
        wa1 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A1 = torch.einsum('bij,bjk->bik', x1, wa1)
        A1 = rearrange(A1, 'b h w -> b w h')  # Transpose
        A1 = A1.softmax(dim=-1)

        VV1 = torch.einsum('bij,bjk->bik', x1, self.token_wV)
        T1 = torch.einsum('bij,bjk->bik', A1, VV1)

        wa2 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A2 = torch.einsum('bij,bjk->bik', x2, wa2)
        A2 = rearrange(A2, 'b h w -> b w h')  # Transpose
        A2 = A2.softmax(dim=-1)

        VV2 = torch.einsum('bij,bjk->bik', x2, self.token_wV)
        T2 = torch.einsum('bij,bjk->bik', A2, VV2)

        cls_tokens1 = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens1, T1), dim=1)
        x1 += self.pos_embedding
        x1 = self.dropout(x1)

        cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_tokens2, T2), dim=1)
        x2 += self.pos_embedding
        x2 = self.dropout(x2)
        
        #print(111111111, x1.shape, x2.shape, z1.shape, z2.shape)
        xx2, z1, z2 = self.corss_fusion(x1, x2, z1, z2)
        xx2, z1, z2 = map(lambda t: t[:, 0], (xx2, z1, z2))### 取第0维分类token        
        
        out_f = self.mlp_head(xx2)# + self.mlp_headz(z1)
        z1_.append(z1)
        z2_.append(z2)

        return [out_f, _, _], [z1_, z2_]