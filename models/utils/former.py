import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.dim_head = dim_head
        self.m = 1
        # self.m_k = nn.Parameter(torch.FloatTensor(1, self.m, heads * dim_head))
        # self.m_v = nn.Parameter(torch.FloatTensor(1, self.m, heads * dim_head))
        
        
        #self.m_k = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_k = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        #self.m_v = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_v = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        torch.nn.init.xavier_normal_(self.m_k)####不用这个初始化，损失函数是NAN
        torch.nn.init.xavier_normal_(self.m_v)

    def forward(self, x):  # 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        b, n, _, h = *x.shape, self.heads
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
                      
                      
        # nk = k.shape[2]### token数
        # m_k = np.sqrt(self.dim_head) * self.m_k.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        # m_v = np.sqrt(self.m) * self.m_v.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        # k = torch.cat([k, m_k], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = torch.cat([v, m_v], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)                      
                      
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        attn = self.attend(dots)  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        # 每个token经过每个head的attention后的输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        return self.to_out(out)


# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        # dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
