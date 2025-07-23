import torch
import torch.nn as nn

class InstanceNorm2d(nn.Module):
    def __init__(self, out_size):
        super(InstanceNorm2d, self).__init__()
        self.norm = nn.InstanceNorm2d(out_size//2, affine=True)

    def forward(self, inputs):

        out_1, out_2 = torch.chunk(inputs, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        return out
        
class InstanceNorm3d(nn.Module):
    def __init__(self, out_size):
        super(InstanceNorm3d, self).__init__()
        self.norm = nn.InstanceNorm3d(out_size//2, affine=True)
        self.bn = nn.BatchNorm3d(out_size)
    def forward(self, inputs):
        if inputs.shape[1]==1:
            out = self.bn(inputs)
        else:
            #print('InstanceNorm3d', inputs.size())
            out_1, out_2 = torch.chunk(inputs, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        return out   