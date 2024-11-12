import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.layers import (
    conv3x3,
    subpel_conv3x3,
    ResidualBlockWithStride,
    ResidualBlock,
    ResidualBlockUpsample,
)


from .compression_bottlenecks import Offset_ELIC, Res_ELIC
from .helpers import (
    FlowNET,
    OffsetDiversity,
    MS_Feature,
    Reconstuctor,
    OffsetTemproalEnc,
    ResidualTemproalEnc,
)


class FlowGuidedB(nn.Module):

    def __init__(self):
        super(FlowGuidedB, self).__init__()

        self.feature_extractor = MS_Feature()

        self.flow_estimator = FlowNET()
        self.offset_temporal_conditioner = OffsetTemproalEnc()

        self.offset_compressor = Offset_ELIC()

        self.offset_diversity_l3 = OffsetDiversity(in_channel=128, magnitude=10)
        self.offset_diversity_l2 = OffsetDiversity(in_channel=96, magnitude=20)
        self.offset_diversity_l1 = OffsetDiversity(in_channel=64, magnitude=40)

        self.residue_temporal_conditioner = ResidualTemproalEnc()
        self.residual_compressor = Res_ELIC()
        self.reconstructor = Reconstuctor()

    def pad_flow(self, tensor):
        (b, c, w, h) = tensor.size()

        p1 = (16 - (w % 16)) % 16
        p2 = (16 - (h % 16)) % 16

        pad = nn.ZeroPad2d(padding=(0, p2, 0, p1))
        return pad(tensor)

    def get_ms_features(self, xref1, xref2, xcur):
        fref1_l1, fref1_l2, fref1_l3 = self.feature_extractor(xref1)
        fref2_l1, fref2_l2, fref2_l3 = self.feature_extractor(xref2)
        fcur_l1, fcur_l2, fcur_l3 = self.feature_extractor(xcur)

        fref1 = [fref1_l1, fref1_l2, fref1_l3]
        fref2 = [fref2_l1, fref2_l2, fref2_l3]
        fcur = [fcur_l1, fcur_l2, fcur_l3]

        return fref1, fref2, fcur

    def convert_scales(self, scale1, scale2, x):
        if torch.is_tensor(scale1) == False:
            scale1 = torch.tensor([scale1])
            scale2 = torch.tensor([scale2])

        scale1 = scale1.view(-1, 1, 1, 1).to(x.device).float()
        scale2 = scale2.view(-1, 1, 1, 1).to(x.device).float()

        scale1 = torch.round(scale1 * 10**2) / (10**2)
        scale2 = torch.round(scale2 * 10**2) / (10**2)

        return scale1, scale2

    def estimate_flow(self, xref1, xref2, down_ratio):

        down_xref1 = F.avg_pool2d(xref1, down_ratio * 2)
        down_xref2 = F.avg_pool2d(xref2, down_ratio * 2)

        (_, _, h, w) = down_xref1.shape
        down_xref1 = self.pad_flow(down_xref1)
        down_xref2 = self.pad_flow(down_xref2)

        flow = self.flow_estimator(torch.cat((down_xref1, down_xref2), dim=1))
        flow = flow[:, :, :h, :w]
        flow = (
            F.interpolate(
                flow, scale_factor=down_ratio, mode="bilinear", align_corners=False
            )
            * down_ratio
        )

        return flow

    def get_warpedrefs_at_layer(self, fref1, fref2, flow, scale1, scale2, layer):
        flow_21, flow_12 = torch.chunk(flow, 2, dim=1)
        flow_cur1 = flow_21 * scale1
        flow_cur2 = flow_12 * scale2

        wref1 = self.warp(fref1, flow_cur1)
        wref2 = self.warp(fref2, flow_cur2)

        down_flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )

        return flow_cur1, flow_cur2, wref1, wref2, down_flow

    def compress_offset(self, wrefs, frefs, fcur, s):
        f_cond_inp_l1 = torch.cat((wrefs[0], wrefs[1], frefs[0], frefs[1]), dim=1)
        f_cond_inp_l2 = torch.cat((wrefs[2], wrefs[3], frefs[2], frefs[3]), dim=1)
        f_cond_inp_l3 = torch.cat((wrefs[4], wrefs[5], frefs[4], frefs[5]), dim=1)

        f_inp_l1 = torch.cat((wrefs[0], wrefs[1], frefs[0], frefs[1], fcur[0]), dim=1)
        f_inp_l2 = torch.cat((wrefs[2], wrefs[3], frefs[2], frefs[3], fcur[1]), dim=1)
        f_inp_l3 = torch.cat((wrefs[4], wrefs[5], frefs[4], frefs[5], fcur[2]), dim=1)

        temporal_condition = self.offset_temporal_conditioner(
            f_cond_inp_l1, f_cond_inp_l2, f_cond_inp_l3
        )

        offset_result = self.offset_compressor(
            f_inp_l1,
            f_inp_l2,
            f_inp_l3,
            f_cond_inp_l1,
            f_cond_inp_l2,
            f_cond_inp_l3,
            temporal_condition,
            s,
        )

        return offset_result

    def get_warped_maps(
        self,
        offset_result,
        flow_cur1_layer,
        flow_cur2_layer,
        fref1_layer,
        fref2_layer,
        offset_diversity,
        layer,
    ):
        offset_hat = offset_result["offset" + str(layer)]
        o1, o2 = torch.chunk(offset_hat, 2, dim=1)

        x = offset_diversity(
            fref1_layer, o1, flow_cur1_layer, fref2_layer, o2, flow_cur2_layer
        )

        return x

    def compress_residue(self, fcur, x_comp_l1, x_comp_l2, x_comp_l3, s):
        temporal_condition = self.residue_temporal_conditioner(
            x_comp_l1, x_comp_l2, x_comp_l3
        )
        residual_result = self.residual_compressor(
            fcur[0],
            fcur[1],
            fcur[2],
            x_comp_l1,
            x_comp_l2,
            x_comp_l3,
            temporal_condition,
            s,
        )

        return residual_result

    def forward(self, xref1, xref2, scale1, scale2, xcur, s, down_ratio):

        B, _, H, W = xcur.shape
        num_pixels = H * W * B

        scale1, scale2 = self.convert_scales(scale1, scale2, xcur)
        flow_l1 = self.estimate_flow(xref1, xref2, down_ratio)
        fref1, fref2, fcur = self.get_ms_features(xref1, xref2, xcur)

        flow_cur1_l1, flow_cur2_l1, wref1_l1, wref2_l1, flow_l2 = (
            self.get_warpedrefs_at_layer(fref1[0], fref2[0], flow_l1, scale1, scale2, 1)
        )

        flow_cur1_l2, flow_cur2_l2, wref1_l2, wref2_l2, flow_l3 = (
            self.get_warpedrefs_at_layer(fref1[1], fref2[1], flow_l2, scale1, scale2, 2)
        )

        flow_cur1_l3, flow_cur2_l3, wref1_l3, wref2_l3, _ = (
            self.get_warpedrefs_at_layer(fref1[2], fref2[2], flow_l3, scale1, scale2, 3)
        )

        wrefs = [wref1_l1, wref2_l1, wref1_l2, wref2_l2, wref1_l3, wref2_l3]
        frefs = [fref1[0], fref2[0], fref1[1], fref2[1], fref1[2], fref2[2]]
        offset_result = self.compress_offset(wrefs, frefs, fcur, s)

        x_comp_l3 = self.get_warped_maps(
            offset_result,
            flow_cur1_l3,
            flow_cur2_l3,
            fref1[2],
            fref2[2],
            self.offset_diversity_l3,
            3,
        )
        x_comp_l2 = self.get_warped_maps(
            offset_result,
            flow_cur1_l2,
            flow_cur2_l2,
            fref1[1],
            fref2[1],
            self.offset_diversity_l2,
            2,
        )
        x_comp_l1 = self.get_warped_maps(
            offset_result,
            flow_cur1_l1,
            flow_cur2_l1,
            fref1[0],
            fref2[0],
            self.offset_diversity_l1,
            1,
        )

        residual_result = self.compress_residue(
            fcur, x_comp_l1, x_comp_l2, x_comp_l3, s
        )

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

    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat(
            [
                flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                flow[:, 1:2, :, :] / ((H - 1.0) / 2.0),
            ],
            1,
        )
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(
            input=img,
            grid=grid_,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return output
