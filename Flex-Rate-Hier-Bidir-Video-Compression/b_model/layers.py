
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models import MeanScaleHyperprior
from compressai.models.utils import conv, deconv
from compressai.layers import (
    GDN,
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


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


class Gain_Module(nn.Module):
        def __init__(self, n=6, N=128, bias=False, inv=False):
            """
            n: number of scales for quantization levels
            N: number of channels
            """
            super(Gain_Module, self).__init__()
            
            self.gain_matrix = nn.Parameter(torch.ones(n, N))
            
            self.bias = bias
            if bias:
                self.bias = nn.Parameter(torch.ones(N))
            
        def forward(self, x, n=None, l=1):
            B, C, H, W = x.shape
            
            # If we want to find a non-trained rate-distortion point
            if (l != 1):
                gain1 = self.gain_matrix[n]
                gain2 = self.gain_matrix[[n[0]+1]]
                gain = (torch.abs(gain1)**l)*(torch.abs(gain2)**(1-l))
                
            else:
                gain = torch.abs(self.gain_matrix[n])
            
            reshaped_gain = gain.unsqueeze(2).unsqueeze(3)
                
            rescaled_latent = reshaped_gain * x
            
            if self.bias:
                rescaled_latent += self.bias[n]
            
            return rescaled_latent
            

class FlowCompressor(MeanScaleHyperprior):

    def __init__(self, n=6, in_ch=19, out_ch=5, N=128, bias=False, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(in_ch, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )
        
        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, out_ch, 2),
        )
        self.g_s[-1][0].weight.data.fill_(0.0)
        self.g_s[-1][0].bias.data.fill_(0.0)
        
        self.gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=False)
        self.inv_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=True)
        
        self.hyper_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=False)
        self.hyper_inv_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=True)
        
    def forward(self, x, n=None, l=None, train=False):
        self.training = train
        
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)

        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings], 
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat
        }
        
        
class ResidualCompressor(MeanScaleHyperprior):

    def __init__(self, n=6, in_ch=3, N=128, bias=False, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(in_ch, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, in_ch, 2),
        )
        
        self.gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=False)
        self.inv_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=True)
        
        self.hyper_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=False)
        self.hyper_inv_gain_unit = Gain_Module(n=n, N=N, bias=bias, inv=True)
        
    def forward(self, x, n=None, l=None, train=False):
        self.training = train
        
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)
        z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    
    def compress(self, x, n, l):
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n, l)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n, l)

        z_strings = self.entropy_bottleneck.compress(scaled_z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)

        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings], 
            "shape": z.size()[-2:]
        }


    def decompress(self, strings, shape, n, l):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scaled_z_hat = self.hyper_inv_gain_unit(z_hat, n, l)
        gaussian_params = self.h_s(scaled_z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        scaled_y_hat = self.inv_gain_unit(y_hat, n, l)
        x_hat = self.g_s(scaled_y_hat).clamp_(0, 1)

        return {
            "x_hat": x_hat
        }
    
    