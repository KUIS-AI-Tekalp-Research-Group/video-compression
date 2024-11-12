import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from compressai.layers import (
    subpel_conv3x3,
    AttentionBlock
)

from .elic import ResidualBottleneckBlock
from torchvision.ops import DeformConv2d


def conv(in_channels, out_channels, kernel_size=5, stride=2, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        groups=groups,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class OffsetDiversity(nn.Module):
    def __init__(self, in_channel, magnitude):
        super().__init__()
        self.in_channel = in_channel
        self.magnitude = magnitude
        self.fusion = DeformConv2d(in_channel*2, in_channel, kernel_size=3, padding=1, groups=2*8)

    def prep(self, out, flow):
        
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)

        offset = torch.tanh(torch.cat((o1, o2), dim=1)) * self.magnitude
        offset = offset + flow.flip(1).repeat(1, offset.size(1)//2, 1, 1)

        return offset, mask
    

    def forward(self, x1, offset1, flow1, x2, offset2, flow2):
        offset1, mask1 = self.prep(offset1, flow1)
        offset2, mask2 = self.prep(offset2, flow2)
        
        x = self.fusion(torch.cat((x1, x2), dim=1), torch.cat((offset1, offset2), dim=1), torch.cat((mask1, mask2), dim=1))

        return x

    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output

    
class MS_Feature(nn.Module):
    def __init__(self):
        super(MS_Feature, self).__init__()                                      
        
        
        self.layer1 = nn.Sequential(
            conv(3, 64, kernel_size=3, stride=2),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
        )
                         
        
        self.layer2 = nn.Sequential(
            conv(64, 96, kernel_size=3, stride=2),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),
        )

        self.layer3 = nn.Sequential(
            conv(96, 128, kernel_size=3, stride=2),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
        )
        
        
                        
    def forward(self, x):
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        return l1, l2, l3


class FlowNET(nn.Module):
    def __init__(self):
        super(FlowNET, self).__init__()
        self.down0 = nn.Sequential(
            conv(6, 32, kernel_size=3, stride=2),
            ResidualBottleneckBlock(32, 32),
            ResidualBottleneckBlock(32, 32),
        )
        self.down1 = nn.Sequential(
            conv(32, 64, kernel_size=3, stride=2),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
        )
        self.down2 = nn.Sequential(
            conv(64, 128, kernel_size=3, stride=2),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
        )
        self.down3 = nn.Sequential(
            conv(128, 192, kernel_size=3, stride=2),
            ResidualBottleneckBlock(192, 192),
            ResidualBottleneckBlock(192, 192),
        )
        self.up0 = nn.Sequential(
            ResidualBottleneckBlock(192, 192),    
            ResidualBottleneckBlock(192, 192),
            subpel_conv3x3(192, 128, 2)
        )
        self.up1 = nn.Sequential(
            conv(256, 128, 1, 1),
            ResidualBottleneckBlock(128, 128),    
            ResidualBottleneckBlock(128, 128),
            subpel_conv3x3(128, 64, 2)
        )
        self.up2 = nn.Sequential(
            conv(128, 64, 1, 1),
            ResidualBottleneckBlock(64, 64),    
            ResidualBottleneckBlock(64, 64),
            subpel_conv3x3(64, 32, 2)
        )
        self.up3 = nn.Sequential(
            conv(64, 32, 1, 1),
            ResidualBottleneckBlock(32, 32),
            ResidualBottleneckBlock(32, 32),
            subpel_conv3x3(32, 4, 2)
        )

    def forward(self, inp):
        s0 = self.down0(inp)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.up0(s3)
        x = self.up1(torch.cat((x, s2), 1)) 
        x = self.up2(torch.cat((x, s1), 1)) 
        x = self.up3(torch.cat((x, s0), 1)) 
        return x
    

class OffsetTemproalEnc(nn.Module):
    def __init__(self, N=128, M=128):
        super(OffsetTemproalEnc, self).__init__()
        
        self.g_a1 = nn.Sequential(conv(64*4, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+96*4, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+128*4, M, kernel_size=5, stride=2),
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
        
        self.g_a1 = nn.Sequential(conv(64, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+96, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+128, M, kernel_size=5, stride=2),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),)
                         
        
    def forward(self, x_comp_l1, x_comp_l2, x_comp_l3):
        
        y = self.g_a1(x_comp_l1)
        y = self.g_a2(torch.cat([y, x_comp_l2], dim=1))
        y = self.g_a3(torch.cat([y, x_comp_l3], dim=1))
        return y



class Reconstuctor(nn.Module):
    def __init__(self):
        super(Reconstuctor, self).__init__()
        
        self.layer3 = nn.Sequential(
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
            ResidualBottleneckBlock(128, 128),
            subpel_conv3x3(128, 128, 2),
        )
        
        self.layer2 = nn.Sequential(
            conv(128+96, 96, 1, 1),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),
            ResidualBottleneckBlock(96, 96),
            subpel_conv3x3(96, 96, 2),
        )
                                      

        self.layer1 = nn.Sequential(
            conv(96+64, 64, 1, 1),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            ResidualBottleneckBlock(64, 64),
            subpel_conv3x3(64, 3, 2),
        )
        
       
            
        
    def forward(self, x_comp_l1, x_comp_l2, x_comp_l3):
        l3 = self.layer3(x_comp_l3)
        l2 = self.layer2(torch.cat([x_comp_l2, l3], dim=1))
        l1 = self.layer1(torch.cat([x_comp_l1, l2], dim=1))
                        
        return l1