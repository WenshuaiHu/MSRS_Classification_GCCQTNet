import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init
#from torchsummary import summary
from models.quaternion_layers import QuaternionLinear
from models.utils.former import FeedForward
import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class CMAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dim_head = dim_head
        self.m = 3
        #self.m_k = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_k = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        #self.m_v = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_v = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        torch.nn.init.xavier_normal_(self.m_k)####不用这个初始化，损失函数是NAN
        torch.nn.init.xavier_normal_(self.m_v)
        
    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross token attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        nk = k.shape[2]### token数
        m_k = np.sqrt(self.dim_head) * self.m_k.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        m_v = np.sqrt(self.m) * self.m_v.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        #print(k.shape, v.shape, m_k.shape, m_v.shape)
        #q = q.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([k, m_k], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([v, m_v], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #print(111111111111, k.shape, v.shape)### 100 8 12 64


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x
        
# cross token attention transformer

class Cross_Attention(nn.Module):
    def __init__(self, h_dim, s_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(h_dim, s_dim, LayerNormalize(s_dim, CMAttention(s_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(s_dim, h_dim, LayerNormalize(h_dim, CMAttention(h_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, h_tokens, l_tokens):
        (h_cls, h_patch_tokens), (l_cls, l_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (h_tokens, l_tokens))
        ###  分别取出两模态的分类token和块token信息
        for h_attend_lg, l_attend_h in self.layers:
            h_cls = h_attend_lg(h_cls, context = l_patch_tokens, kv_include_self = True) + h_cls
            l_cls = l_attend_h(l_cls, context = h_patch_tokens, kv_include_self = True) + l_cls

        h_tokens = torch.cat((h_cls, h_patch_tokens), dim = 1)
        l_tokens = torch.cat((l_cls, l_patch_tokens), dim = 1)
        return h_tokens, l_tokens
        
class Quater_Cross_Transformer(nn.Module):
    def __init__(self, h_dim, depth, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormalize(4*h_dim, QuaternionLinear(4*h_dim, 4*h_dim)),
                LayerNormalize(4*h_dim, FeedForward(4*h_dim, 4*h_dim*2, dropout=dropout))
            ]))
            
    def forward(self, h_tokens, l_tokens, zh_tokens, zl_tokens):
        
        ps = []
        ps.append(h_tokens)
        ps.append(l_tokens)
        ps.append(zh_tokens)
        ps.append(zl_tokens)
        tokens = torch.cat(ps, dim=2)#
        
        for attn, ff in self.layers:
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens
        output_tokens = tokens
        return output_tokens
        
# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        
class Attention_3_branches(nn.Module):
    def __init__(self, emb_dropout, feature_dim, ct_attn_depth = 1, ct_attn_heads = 8):
        super().__init__()
        ct_attn_dim_head = feature_dim
        h_dim = feature_dim
        s_dim = feature_dim
        z_dim = feature_dim
        self.CThz = Cross_Attention(h_dim = h_dim, s_dim = z_dim, depth = ct_attn_depth, 
                        heads = ct_attn_heads, dim_head = ct_attn_dim_head, dropout = emb_dropout)
        self.CThs = Cross_Attention(h_dim = h_dim, s_dim = s_dim, depth = ct_attn_depth, 
                        heads = ct_attn_heads, dim_head = ct_attn_dim_head, dropout = emb_dropout)
        self.out = Quater_Cross_Transformer(h_dim, ct_attn_depth, emb_dropout)
        
    def forward(self, x, x2, z, z2):
        xz, z = self.CThz(x, z)###x和z交互
        x2z, z2 = self.CThz(x2, z2)###x2和z2交互
        xx2, x2x = self.CThs(x, x2)###x和x2交互
        xx2 = self.out(xz, x2z, xx2, x2x)

        return xx2, z, z2
        