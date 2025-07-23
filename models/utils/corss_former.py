import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init
from torchsummary import summary
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
        
        '''
        nk = k.shape[2]### token数
        m_k = np.sqrt(self.dim_head) * self.m_k.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        m_v = np.sqrt(self.m) * self.m_v.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        #print(k.shape, v.shape, m_k.shape, m_v.shape)
        #q = q.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([k, m_k], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([v, m_v], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #print(111111111111, k.shape, v.shape)### 100 8 12 64'''


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        
# cross token attention transformer

class Cross_Attention(nn.Module):
    def __init__(self, h_dim, s_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormalize(s_dim, CMAttention(s_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                LayerNormalize(h_dim, CMAttention(h_dim, heads = heads, dim_head = dim_head, dropout = dropout))
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