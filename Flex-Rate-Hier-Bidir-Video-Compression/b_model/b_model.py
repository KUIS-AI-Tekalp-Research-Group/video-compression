
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.deform_conv as df

import math

from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.layers import GDN

from .layers import FlowCompressor, ResidualCompressor
from .unet import UNet
           
device = torch.device("cuda")


class BidirFlowRef(nn.Module):
    """
    Bidirectional Compression with Flow Refinement
    """
    def __init__(self, n=6, N=128):
        super(BidirFlowRef, self).__init__()
        
        self.flow_predictor = UNet(6, 4, 5)
        self.Mask = UNet(16,2,4)
        
        self.flow_compressor = FlowCompressor(n=n, in_ch=19, out_ch=4, N=N, bias=False)
        self.residual_compressor = ResidualCompressor(n=n, in_ch=3, N=N, bias=False)
        
        
    def process(self, x0, x1, t=0.5):
        x = torch.cat((x0, x1), 1)
        Flow = self.flow_predictor(x)
        Flow_0_1, Flow_1_0 = Flow[:,:2,:,:], Flow[:,2:4,:,:]
        Flow_t_0 = -(1-t)*t*Flow_0_1+t*t*Flow_1_0
        Flow_t_1 = (1-t)*(1-t)*Flow_0_1-t*(1-t)*Flow_1_0
        
        xt1 = self.backwarp(x0, Flow_t_0)
        xt2 = self.backwarp(x1, Flow_t_1)
        
        return Flow_t_0, Flow_t_1, torch.cat((Flow_t_0,Flow_t_1,x,xt1,xt2),1)
        
        
        
    def forward(self, x_before, x_current, x_after, n=None, l=1, train=False):
        x = torch.cat((x_before, x_after), 1)
        
        _, _, H, W = x_current.shape
        num_pixels = H * W
        
        mv_before, mv_after, x_conc = self.process(x_before, x_after)
        x_input = torch.cat((x_conc, x_current), 1)
        
        flow_result = self.flow_compressor(x_input, n, l, train)
        flow_hat = flow_result["x_hat"]
        
        mv_before_refined = mv_before + flow_hat[:, :2, :, :]
        mv_after_refined = mv_after + flow_hat[:, 2:4, :, :]
        
        x_b = self.backwarp(x_before, mv_before_refined)
        x_a = self.backwarp(x_after, mv_after_refined)
        
        temp = torch.cat((mv_before_refined,mv_after_refined,x,x_b,x_a),1)
        mask = F.sigmoid(self.Mask(temp))
        
        w1, w2 = 0.5*mask[:,0:1,:,:], 0.5*mask[:,1:2,:,:]
        x_comp = (w1*x_b+w2*x_a)/(w1+w2+1e-8)
                        
        residual = x_current - x_comp
        
        residual_result = self.residual_compressor(residual, n, l, train)
        residual_hat = residual_result["x_hat"]
        
        x_hat = x_comp + residual_hat
        
        size_flow = sum(
            (torch.log(likelihoods).sum(dim=(1, 2, 3)) / (-math.log(2)))
            for likelihoods in flow_result["likelihoods"].values()
        )
        rate_flow = size_flow / num_pixels
        
        size_residual = sum(
            torch.log(likelihoods).sum(dim=(1, 2, 3)) / (-math.log(2))
            for likelihoods in residual_result["likelihoods"].values()
        )
        rate_residual = size_residual / num_pixels
        
        return {
            "x_hat": x_hat,
            "size": size_flow + size_residual,
            "rate": rate_flow + rate_residual
        }
        
    
    def backwarp(self, img, flow):
        _, _, H, W = img.size()
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        gridX = torch.tensor(gridX, requires_grad=False).cuda()
        gridY = torch.tensor(gridY, requires_grad=False).cuda()
        u = flow[:,0,:,:]
        v = flow[:,1,:,:]
        x = gridX.unsqueeze(0).expand_as(u).float()+u
        y = gridY.unsqueeze(0).expand_as(v).float()+v
        normx = 2*(x/W-0.5)
        normy = 2*(y/H-0.5)
        grid = torch.stack((normx,normy), dim=3)
        warped = F.grid_sample(img, grid)
        return warped