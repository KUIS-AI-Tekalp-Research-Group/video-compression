import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from compressai.models import Cheng2020Anchor, JointAutoregressiveHierarchicalPriors
from .layers import CheckerboardContext
from torch import Tensor
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from .elic import ResidualBottleneckBlock

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def ste_round(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.
    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_
    .. note::
        Implemented with the pytorch `detach()` reparametrization trick:
        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
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




class Offset_ELIC(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, M=128, **kwargs):
        super().__init__(N, M, **kwargs)

        
        self.g_a1 = nn.Sequential(conv(64*5, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+96*5, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+128*5, M, kernel_size=5, stride=2),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),)
        

        self.g_s3 = nn.Sequential(
            ResidualBottleneckBlock(M, M),
            ResidualBottleneckBlock(M, M),
            ResidualBottleneckBlock(M, M),
            deconv(M, N, kernel_size=5, stride=2),)

        self.g_o3 = nn.Sequential(
            conv(N+128*4, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 27*8*2, kernel_size=3, stride=1))
        
        self.g_s2 = nn.Sequential(
            conv(N+128*4, N, kernel_size=1, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),)
        
        self.g_o2 = nn.Sequential(
            conv(N+96*4, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 27*8*2, kernel_size=3, stride=1))
        
        self.g_s1 = nn.Sequential(
            conv(N+96*4, N, kernel_size=1, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),)
        
        self.g_o1 = nn.Sequential(
            conv(N+64*4, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 27*8*2, kernel_size=3, stride=1))
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            deconv(M, M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(M, M, stride=1, kernel_size=3),
        )

        self.prior_fusion = nn.Sequential(
            conv(2*M, 2*M, stride=1, kernel_size=3),
            ResidualBottleneckBlock(2*M, 2*M),
            ResidualBottleneckBlock(2*M, 2*M),
            ResidualBottleneckBlock(2*M, 2*M),
            conv(M*2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channel, M * 10 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(M * 8 // 3, out_channel * 6 // 3, 1),
                )
                for in_channel, out_channel in [
                    (M * 4, 6),
                    (M * 6, 6),
                    (M * 6, 12),
                    (M * 6, 24),
                    (M * 6, M - 48),
                ]
            ]
        )

        self.channel_context_models = nn.ModuleList(
            [
                nn.Sequential(
                    conv(in_channels, N, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, M * 2, kernel_size=5, stride=1),
                )
                for in_channels in [6, 12, 24, 48]
            ]
        )

        self.context_prediction_models = nn.ModuleList(
            [
                CheckerboardContext(
                    in_channels=in_channel,
                    out_channels=M * 2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
                for in_channel in [6, 6, 12, 24, M - 48]
            ]
        )
        
        
        levels = 5
        self.levels = levels
    
        self.Gain = torch.nn.Parameter(torch.ones(size=[levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[levels, N]), requires_grad=True)


    def forward(self, f1, f2, f3, f1d, f2d, f3d, offset_temp, s):
        gain, hypergain, invhypergain, invgain = self.interpolate_gain(s)
        
        y = self.g_a1(f1)
        y = self.g_a2(torch.cat([y, f2], dim=1))
        y = self.g_a3(torch.cat([y, f3], dim=1))
        y = y * gain.unsqueeze(0).unsqueeze(2).unsqueeze(3) 
        z = self.h_a(y)
        z = z * hypergain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        likelihoods_list = {}
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = ste_round(z)
        z_hat = z_hat * invhypergain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        hyper_params = self.h_s(z_hat)
        hyper_params = self.prior_fusion(torch.cat([hyper_params, offset_temp], dim=1))
        likelihoods_list["z"] = z_likelihoods
        uneven_groups = [
            y[:, :6, :, :],
            y[:, 6:12, :, :],
            y[:, 12:24, :, :],
            y[:, 24:48, :, :],
            y[:, 48:, :, :],
        ]

        for i, curr_y in enumerate(uneven_groups):
            curr_y_hat = ste_round(curr_y)

            y_half = curr_y_hat.clone()
            y_half[:, :, 0::2, 0::2] = 0
            y_half[:, :, 1::2, 1::2] = 0

            ctx_params = self.context_prediction_models[i](y_half)
            ctx_params[:, :, 0::2, 1::2] = 0
            ctx_params[:, :, 1::2, 0::2] = 0
            if i == 0:
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                likelihoods_list[f"y_{i}"] = y_likelihoods
            else:
                channel_context_in = ste_round(torch.cat(uneven_groups[:i], dim=1))
                channel_context = self.channel_context_models[i - 1](channel_context_in)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, channel_context, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                likelihoods_list[f"y_{i}"] = y_likelihoods

        y_hat_ = ste_round(y)
        y_hat = y_hat_ * invgain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        xhat3 = self.g_s3(y_hat)
        inp3 = torch.cat([xhat3, f3d], dim=1)
        offset3 = self.g_o3(inp3)
        xhat2 = self.g_s2(inp3)
        inp2 = torch.cat([xhat2, f2d], dim=1)
        offset2 = self.g_o2(inp2)
        xhat1 = self.g_s1(inp2)
        inp1 = torch.cat([xhat1, f1d], dim=1)
        offset1 = self.g_o1(inp1)

        
        return {
            "offset3": offset3,
            "offset2": offset2,
            "offset1": offset1,
            "likelihoods": likelihoods_list,
        }
    
    
    def interpolate_gain(self, s):
        s = min(s, self.levels-1)
        s = max(s, 0)
        upper = int(min(math.ceil(s), self.levels-1))
        lower = int(max(math.floor(s), 0))

        if upper == lower:
            s = int(s)
            gain = torch.abs(self.Gain[s])
            hypergain = torch.abs(self.HyperGain[s])
            invhypergain = torch.abs(self.InverseHyperGain[s])
            invgain = torch.abs(self.InverseGain[s])

        else:
            l = upper - s
            gain = torch.abs(self.Gain[upper])**(1-l) * torch.abs(self.Gain[lower])**(l)
            hypergain = torch.abs(self.HyperGain[upper])**(1-l) * torch.abs(self.HyperGain[lower])**(l)
            invhypergain = torch.abs(self.InverseHyperGain[upper])**(1-l) * torch.abs(self.InverseHyperGain[lower])**(l)
            invgain = torch.abs(self.InverseGain[upper])**(1-l) * torch.abs(self.InverseGain[lower])**(l)

        return gain, hypergain, invhypergain, invgain
    

class Res_ELIC(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, M=128, **kwargs):
        super().__init__(N, M, **kwargs)

        
        self.g_a1 = nn.Sequential(
            conv(64*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a2 = nn.Sequential(conv(N+96*2, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),)
        
        self.g_a3 = nn.Sequential(conv(N+128*2, M, kernel_size=5, stride=2),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),
                                  ResidualBottleneckBlock(M, M),)
        

        self.g_s3 = nn.Sequential(
            ResidualBottleneckBlock(M, M),
            ResidualBottleneckBlock(M, M),
            ResidualBottleneckBlock(M, M),
            deconv(M, N, kernel_size=5, stride=2),)
        self.g_o3 = nn.Sequential(
            conv(N+128, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 128, kernel_size=3, stride=1))
        
        self.g_s2 = nn.Sequential(
            conv(N+128, N, kernel_size=1, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),)
        
        self.g_o2 = nn.Sequential(
            conv(N+96, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 96, kernel_size=3, stride=1))
        
        self.g_s1 = nn.Sequential(
            conv(N+96, N, kernel_size=1, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),)
        
        self.g_o1 = nn.Sequential(
            conv(N+64, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, 64, kernel_size=3, stride=1))
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            deconv(M, M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(M, M, stride=1, kernel_size=3),
        )

        self.prior_fusion = nn.Sequential(
            conv(2*M, 2*M, stride=1, kernel_size=3),
            ResidualBottleneckBlock(2*M, 2*M),
            ResidualBottleneckBlock(2*M, 2*M),
            ResidualBottleneckBlock(2*M, 2*M),
            conv(M*2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channel, M * 10 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(M * 8 // 3, out_channel * 6 // 3, 1),
                )
                for in_channel, out_channel in [
                    (M * 4, 6),
                    (M * 6, 6),
                    (M * 6, 12),
                    (M * 6, 24),
                    (M * 6, M - 48),
                ]
            ]
        )

        self.channel_context_models = nn.ModuleList(
            [
                nn.Sequential(
                    conv(in_channels, N, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, M * 2, kernel_size=5, stride=1),
                )
                for in_channels in [6, 12, 24, 48]
            ]
        )

        self.context_prediction_models = nn.ModuleList(
            [
                CheckerboardContext(
                    in_channels=in_channel,
                    out_channels=M * 2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
                for in_channel in [6, 6, 12, 24, M - 48]
            ]
        )
        
        
        levels = 5
        self.levels = levels
    
        self.Gain = torch.nn.Parameter(torch.ones(size=[levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[levels, N]), requires_grad=True)


    def forward(self, f1, f2, f3, f1d, f2d, f3d, residual_temp, s):
        gain, hypergain, invhypergain, invgain = self.interpolate_gain(s)
        
        y = self.g_a1(torch.cat([f1, f1d], dim=1))
        y = self.g_a2(torch.cat([y, f2, f2d], dim=1))
        y = self.g_a3(torch.cat([y, f3, f3d], dim=1))
        y = y * gain.unsqueeze(0).unsqueeze(2).unsqueeze(3) 

        z = self.h_a(y)
        z = z * hypergain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        likelihoods_list = {}
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = ste_round(z)
        z_hat = z_hat * invhypergain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        hyper_params = self.h_s(z_hat)
        hyper_params = self.prior_fusion(torch.cat([hyper_params, residual_temp], dim=1))
        likelihoods_list["z"] = z_likelihoods
        uneven_groups = [
            y[:, :6, :, :],
            y[:, 6:12, :, :],
            y[:, 12:24, :, :],
            y[:, 24:48, :, :],
            y[:, 48:, :, :],
        ]

        for i, curr_y in enumerate(uneven_groups):
            curr_y_hat = ste_round(curr_y)

            y_half = curr_y_hat.clone()
            y_half[:, :, 0::2, 0::2] = 0
            y_half[:, :, 1::2, 1::2] = 0

            ctx_params = self.context_prediction_models[i](y_half)
            ctx_params[:, :, 0::2, 1::2] = 0
            ctx_params[:, :, 1::2, 0::2] = 0
            if i == 0:
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                likelihoods_list[f"y_{i}"] = y_likelihoods
            else:
                channel_context_in = ste_round(torch.cat(uneven_groups[:i], dim=1))
                channel_context = self.channel_context_models[i - 1](channel_context_in)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, channel_context, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                likelihoods_list[f"y_{i}"] = y_likelihoods

        y_hat_ = ste_round(y)
        y_hat = y_hat_ * invgain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        xhat3 = self.g_s3(y_hat)
        inp3 = torch.cat([xhat3, f3d], dim=1)
        res3 = self.g_o3(inp3)
        xhat2 = self.g_s2(inp3)
        inp2 = torch.cat([xhat2, f2d], dim=1)
        res2 = self.g_o2(inp2)
        xhat1 = self.g_s1(inp2)
        inp1 = torch.cat([xhat1, f1d], dim=1)
        res1 = self.g_o1(inp1)
        
        return {
            "res3": res3,
            "res2": res2,
            "res1": res1,
            "likelihoods": likelihoods_list,
        }
    
    
    def interpolate_gain(self, s):
        s = min(s, self.levels-1)
        s = max(s, 0)
        upper = int(min(math.ceil(s), self.levels-1))
        lower = int(max(math.floor(s), 0))

        if upper == lower:
            s = int(s)
            gain = torch.abs(self.Gain[s])
            hypergain = torch.abs(self.HyperGain[s])
            invhypergain = torch.abs(self.InverseHyperGain[s])
            invgain = torch.abs(self.InverseGain[s])

        else:
            l = upper - s
            gain = torch.abs(self.Gain[upper])**(1-l) * torch.abs(self.Gain[lower])**(l)
            hypergain = torch.abs(self.HyperGain[upper])**(1-l) * torch.abs(self.HyperGain[lower])**(l)
            invhypergain = torch.abs(self.InverseHyperGain[upper])**(1-l) * torch.abs(self.InverseHyperGain[lower])**(l)
            invgain = torch.abs(self.InverseGain[upper])**(1-l) * torch.abs(self.InverseGain[lower])**(l)

        return gain, hypergain, invhypergain, invgain
    