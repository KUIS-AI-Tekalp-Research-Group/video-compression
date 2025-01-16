
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

import math

from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.layers import conv3x3, subpel_conv3x3, ResidualBlockWithStride, ResidualBlock, ResidualBlockUpsample


from .offset_res_elic import Offset_ELIC, Res_ELIC
from .helpers import MS_Feature, ResidualTemproalEnc, OffsetTemproalEnc, Reconstuctor
           

class DeformB(nn.Module):
    def __init__(self):
        super(DeformB, self).__init__()

        self.feature_extractor = MS_Feature()
        
        self.offset_temp_encoder = OffsetTemproalEnc()
        self.offset_compressor = Offset_ELIC()

        self.deconv_l3_1 = DeformConv2d(96, 96, kernel_size=3, padding=1, groups=8)
        self.deconv_l3_2 = DeformConv2d(96, 96, kernel_size=3, padding=1, groups=8)
        self.deconv_l2_1 = DeformConv2d(64, 64, kernel_size=3, padding=1, groups=8)
        self.deconv_l2_2 = DeformConv2d(64, 64, kernel_size=3, padding=1, groups=8)
        self.deconv_l1_1 = DeformConv2d(32, 32, kernel_size=3, padding=1, groups=8)
        self.deconv_l1_2 = DeformConv2d(32, 32, kernel_size=3, padding=1, groups=8)
                
        self.reconstructor = Reconstuctor()
    
        self.residual_temp_encoder = ResidualTemproalEnc()
        self.residual_compressor = Res_ELIC()


    def get_ms_features(self, xref1, xref2, xcur):
        fref1_l1, fref1_l2, fref1_l3 = self.feature_extractor(xref1)
        fref2_l1, fref2_l2, fref2_l3 = self.feature_extractor(xref2)
        fcur_l1, fcur_l2, fcur_l3 = self.feature_extractor(xcur)
        
        fref1 = [fref1_l1, fref1_l2, fref1_l3]
        fref2 = [fref2_l1, fref2_l2, fref2_l3]
        fcur = [fcur_l1, fcur_l2, fcur_l3]
        
        return fref1, fref2, fcur
    
    
    
    def compress_offset(self, frefs, fcur, s):
        f_cond_inp_l1 = torch.cat((frefs[0], frefs[1]), dim=1)
        f_cond_inp_l2 = torch.cat((frefs[2], frefs[3]), dim=1)
        f_cond_inp_l3 = torch.cat((frefs[4], frefs[5]), dim=1)
        
        f_inp_l1 = torch.cat((frefs[0], frefs[1], fcur[0]), dim=1)
        f_inp_l2 = torch.cat((frefs[2], frefs[3], fcur[1]), dim=1)
        f_inp_l3 = torch.cat((frefs[4], frefs[5], fcur[2]), dim=1)
        

        offset_temp = self.offset_temp_encoder(f_cond_inp_l1, f_cond_inp_l2, f_cond_inp_l3)
        offset_result = self.offset_compressor(f_inp_l1, f_inp_l2, f_inp_l3, f_cond_inp_l1, 
                                               f_cond_inp_l2, f_cond_inp_l3, offset_temp, s)
        
        return offset_result
    

    def get_deformed_maps(self, offset_result, fref1_layer, fref2_layer, deconv_layer_1, deconv_layer_2, layer):
        offset_hat = offset_result["offset"+str(layer)]
        o1, o2 = torch.chunk(offset_hat, 2, dim=1)
        o1x, o1y, m1 = torch.chunk(o1, 3, dim=1)
        o2x, o2y, m2 = torch.chunk(o2, 3, dim=1)

        o1 = torch.cat((o1x, o1y), dim=1)
        o2 = torch.cat((o2x, o2y), dim=1)
        
        m1 = torch.sigmoid(m1)
        m2 = torch.sigmoid(m2)

        x1 = deconv_layer_1(fref1_layer, o1, m1)
        x2 = deconv_layer_2(fref2_layer, o2, m2)
        return torch.cat([x1, x2], dim=1)
     

    def compress_residue(self, xcur, fcur, x_comp_l1, x_comp_l2, x_comp_l3, s):
        
        residual_temp = self.residual_temp_encoder(x_comp_l1, x_comp_l2, x_comp_l3)
        residual_result = self.residual_compressor(xcur, fcur[0],fcur[1],fcur[2], x_comp_l1, x_comp_l2, x_comp_l3, residual_temp, s)
        
        return residual_result
    
    def forward(self, xref1, xref2, xcur, s):
        
        B, _, H, W = xcur.shape
        num_pixels = H * W * B
        
        fref1, fref2, fcur = self.get_ms_features(xref1, xref2, xcur)
        
        frefs = [fref1[0], fref2[0], fref1[1], fref2[1], fref1[2], fref2[2]]
        offset_result = self.compress_offset(frefs, fcur, s)

        
        x_comp_l3 = self.get_deformed_maps(offset_result, fref1[2], fref2[2], self.deconv_l3_1, self.deconv_l3_2, 3)
        x_comp_l2 = self.get_deformed_maps(offset_result, fref1[1], fref2[1], self.deconv_l2_1, self.deconv_l2_2, 2)
        x_comp_l1 = self.get_deformed_maps(offset_result, fref1[0], fref2[0], self.deconv_l1_1, self.deconv_l1_2, 1)
                

        residual_result = self.compress_residue(xcur, fcur, x_comp_l1, x_comp_l2, x_comp_l3, s)
        
        
        x_comp_l3 = x_comp_l3 + residual_result["res3"]
        x_comp_l2 = x_comp_l2 + residual_result["res2"]
        x_comp_l1 = x_comp_l1 + residual_result["res1"]
        
        x_hat = self.reconstructor(x_comp_l1, x_comp_l2, x_comp_l3)
        
        size_offset = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in offset_result["likelihoods"].values()
        )
        rate_offset = size_offset / num_pixels
        
        size_residual = sum(
            (torch.log(likelihoods).sum() / (-math.log(2)))
            for likelihoods in residual_result["likelihoods"].values()
        )
        rate_residual = size_residual / num_pixels

        return {
            "x_hat": x_hat,
            "size": size_offset + size_residual,
            "rate": rate_offset + rate_residual,
        }

   