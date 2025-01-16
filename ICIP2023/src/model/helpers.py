
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.utils import conv, deconv
from compressai.layers import subpel_conv3x3


from .elic import ResidualBottleneckBlock


class MS_Feature(nn.Module):
    def __init__(self):
        super(MS_Feature, self).__init__()                                      
        
        
        self.layer1 = nn.Sequential(
            conv(3, 32, kernel_size=3, stride=2),
            ResidualBottleneckBlock(32, 32),
            ResidualBottleneckBlock(32, 32),    
            ResidualBottleneckBlock(32, 32),
        )
                         

        self.layer2 = nn.Sequential(
            conv(32, 64, kernel_size=3, stride=2),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),

        )
        
        self.layer3 = nn.Sequential(
            conv(64, 96, kernel_size=3, stride=2),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),

        )
        
        
                        
    def forward(self, x):
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        return l1, l2, l3


class Reconstuctor(nn.Module):
    def __init__(self):
        super(Reconstuctor, self).__init__()
        
        self.layer3 = nn.Sequential(
            ResidualBottleneckBlock(192, 192),
            ResidualBottleneckBlock(192, 192),
            ResidualBottleneckBlock(192, 192),
            deconv(192, 192, stride=2, kernel_size=3),
        )
        
        self.layer2 = nn.Sequential(
            conv(192+128, 128, 1, 1),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
            deconv(128, 128, stride=2, kernel_size=3),
        )
                                      

        self.layer1 = nn.Sequential(
            conv(128+64, 64, 1, 1),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            deconv(64, 3, stride=2, kernel_size=3),
        )
        
       
            
        
    def forward(self, x_comp_l1, x_comp_l2, x_comp_l3):
        l3 = self.layer3(x_comp_l3)
        l2 = self.layer2(torch.cat([x_comp_l2, l3], dim=1))
        l1 = self.layer1(torch.cat([x_comp_l1, l2], dim=1))
                        
        return l1
    
    
class OffsetTemproalEnc(nn.Module):
    def __init__(self, N=128, M=128):
        super(OffsetTemproalEnc, self).__init__()
        
        self.g_a1 = nn.Sequential(conv(32*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+64*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+96*2, M, kernel_size=5, stride=2),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),)
                         
        
    def forward(self, f_cond_inp_l1, f_cond_inp_l2, f_cond_inp_l3):
        
        y = self.g_a1(f_cond_inp_l1)
        y = self.g_a2(torch.cat([y, f_cond_inp_l2], dim=1))
        y = self.g_a3(torch.cat([y, f_cond_inp_l3], dim=1))
        return y
    
    
class ResidualTemproalEnc(nn.Module):
    def __init__(self, N=128, M=128):
        super(ResidualTemproalEnc, self).__init__()
        
        self.g_a1 = nn.Sequential(conv(32*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+64*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+96*2, M, kernel_size=5, stride=2),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),)
                         
        
    def forward(self, x_comp_l1, x_comp_l2, x_comp_l3):
        
        y = self.g_a1(x_comp_l1)
        y = self.g_a2(torch.cat([y, x_comp_l2], dim=1))
        y = self.g_a3(torch.cat([y, x_comp_l3], dim=1))
        return y