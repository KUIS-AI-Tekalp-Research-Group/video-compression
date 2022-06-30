import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import ScaleHyperprior
from compressai.models.utils import conv, deconv

device = torch.device("cuda")

class Gain_Module(nn.Module):
        def __init__(self, n=6, N=128, inv=False):
            """
            n: number of scales for quantization levels
            N: number of channels
            """
            super(Gain_Module, self).__init__()
            
            self.gain_matrix = nn.Parameter(torch.ones(n, N))

        def forward(self, x, n=None):
            l = n - n.long()
            n = n.long()
            
            # If we want to find a non-trained rate-distortion point
            if (torch.any(l != 0)):
                gain1 = self.gain_matrix[n]
                gain2 = self.gain_matrix[n+1]
                gain = (torch.abs(gain1)**(1-l))*(torch.abs(gain2)**l)
                
            else:
                gain = torch.abs(self.gain_matrix[n])
            
            reshaped_gain = gain.unsqueeze(2).unsqueeze(3)
            rescaled_latent = reshaped_gain * x
            return rescaled_latent


class gained_mbt2018(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
            
    https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/models/google.py
    """

    def __init__(self, N, M, n, metric="mse", **kwargs):
        super().__init__(N, M, **kwargs)
        
        self.metric = metric

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        
        self.gain_unit = Gain_Module(n=n, N=M, inv=False)
        self.inv_gain_unit = Gain_Module(n=n, N=M, inv=True)
        
        self.hyper_gain_unit = Gain_Module(n=n, N=N, inv=False)
        self.hyper_inv_gain_unit = Gain_Module(n=n, N=N, inv=True)
        
        self.init_shapes()
    
    def init_shapes(self):
        self.entropy_bottleneck._offset = torch.zeros((192)).to(device).type(torch.int32)
        if self.metric == "mse":
            self.entropy_bottleneck._quantized_cdf = torch.zeros((192, 183)).to(device).type(torch.int32)
        elif self.metric == "ms-ssim":
            self.entropy_bottleneck._quantized_cdf = torch.zeros((192, 133)).to(device).type(torch.int32)
        self.entropy_bottleneck._cdf_length = torch.zeros((192)).to(device).type(torch.int32)
        self.gaussian_conditional._offset = torch.zeros((64)).to(device).type(torch.int32)
        self.gaussian_conditional._quantized_cdf = torch.zeros((64, 3133)).to(device).type(torch.int32)
        self.gaussian_conditional._cdf_length = torch.zeros((64)).to(device).type(torch.int32)
        self.gaussian_conditional.scale_table = torch.zeros((64)).to(device).type(torch.float32)

    def forward(self, x, n):
        
        y = self.g_a(x)
        scaled_y = self.gain_unit(y, n)
        z = self.h_a(scaled_y)
        scaled_z = self.hyper_gain_unit(z, n)
        scaled_z_hat, z_likelihoods = self.entropy_bottleneck(scaled_z)
        z_hat = self.hyper_inv_gain_unit(scaled_z_hat, n)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        scaled_y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, scales_hat, means=means_hat)
        y_hat = self.inv_gain_unit(scaled_y_hat, n)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


