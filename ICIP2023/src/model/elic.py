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


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.BottleneckBlock = nn.Sequential(
            conv1x1(in_ch, out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.ReLU(inplace=True),
            conv1x1(out_ch, out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.BottleneckBlock(x)
        out = out + identity
        return out


class ELIC(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

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
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
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
                    (M * 4, 16),
                    (M * 6, 16),
                    (M * 6, 32),
                    (M * 6, 64),
                    (M * 6, M - 128),
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
                for in_channels in [16, 32, 64, 128]
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
                for in_channel in [16, 16, 32, 64, M - 128]
            ]
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        likelihoods_list = {}
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = ste_round(z)
        hyper_params = self.h_s(z_hat)
        likelihoods_list["z"] = z_likelihoods
        uneven_groups = [
            y[:, :16, :, :],
            y[:, 16:32, :, :],
            y[:, 32:64, :, :],
            y[:, 64:128, :, :],
            y[:, 128:, :, :],
        ]

        for i, curr_y in enumerate(uneven_groups):
            curr_y_hat = self.gaussian_conditional.quantize(
                curr_y, "noise" if self.training else "dequantize"
            )

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
                channel_context_in = self.gaussian_conditional.quantize(
                    torch.cat(uneven_groups[:i], dim=1),
                    "noise" if self.training else "dequantize",
                )
                channel_context = self.channel_context_models[i - 1](channel_context_in)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, channel_context, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                likelihoods_list[f"y_{i}"] = y_likelihoods

        y_hat = ste_round(y)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods_list,
        }
    
    def forward_stage2(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        likelihoods_list = {}
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = ste_round(z)
        hyper_params = self.h_s(z_hat)
        likelihoods_list["z"] = z_likelihoods
        uneven_groups = [
            y[:, :16, :, :],
            y[:, 16:32, :, :],
            y[:, 32:64, :, :],
            y[:, 64:128, :, :],
            y[:, 128:, :, :],
        ]

        for i, curr_y in enumerate(uneven_groups):
            curr_y_hat = self.gaussian_conditional.quantize(
                curr_y, "noise" if self.training else "dequantize"
            )

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
                uneven_groups[i] = ste_round(curr_y-means_hat)+means_hat
                likelihoods_list[f"y_{i}"] = y_likelihoods
            else:
                channel_context_in = self.gaussian_conditional.quantize(
                    torch.cat(uneven_groups[:i], dim=1),
                    "noise" if self.training else "dequantize",
                )
                channel_context = self.channel_context_models[i - 1](channel_context_in)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, channel_context, hyper_params), dim=1)
                )
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                _, y_likelihoods = self.gaussian_conditional(
                    curr_y, scales_hat, means=means_hat
                )
                uneven_groups[i] = ste_round(curr_y-means_hat)+means_hat
                likelihoods_list[f"y_{i}"] = y_likelihoods

        y_hat = torch.cat(uneven_groups,dim=1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods_list,
        }

    def compress(self, x):
        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        y_strings_all = []
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)

        uneven_groups = [
            y[:, :16, :, :],
            y[:, 16:32, :, :],
            y[:, 32:64, :, :],
            y[:, 64:128, :, :],
            y[:, 128:, :, :],
        ]

        for i, curr_y in enumerate(uneven_groups):
            encoder = BufferedRansEncoder()
            symbols_list = []
            indexes_list = []
            y_strings = []
            ctx_params_anchor = torch.zeros(
                [curr_y.size(0), 320 * 2, curr_y.size(2), curr_y.size(3)],
                device=curr_y.device,
            )

            if i == 0:
                gaussian_params_anchor = self.entropy_parameters[i](
                    torch.cat([ctx_params_anchor, hyper_params], dim=1)
                )

                scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
                anchor_hat = self.compress_anchor(
                    curr_y, scales_anchor, means_anchor, symbols_list, indexes_list
                )
                ctx_params = self.context_prediction_models[i](anchor_hat)

                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, hyper_params), dim=1)
                )
                scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
                nonanchor_hat = self.compress_nonanchor(
                    curr_y,
                    scales_nonanchor,
                    means_nonanchor,
                    symbols_list,
                    indexes_list,
                )

                uneven_groups[i] = anchor_hat + nonanchor_hat

            else:
                channel_context = self.channel_context_models[i - 1](
                    torch.cat(uneven_groups[:i], dim=1)
                )

                gaussian_params_anchor = self.entropy_parameters[i](
                    torch.cat(
                        [
                            ctx_params_anchor,
                            channel_context,
                            hyper_params,
                        ],
                        dim=1,
                    )
                )

                scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
                anchor_hat = self.compress_anchor(
                    curr_y, scales_anchor, means_anchor, symbols_list, indexes_list
                )
                ctx_params = self.context_prediction_models[i](anchor_hat)

                gaussian_params = self.entropy_parameters[i](
                    torch.cat((ctx_params, channel_context, hyper_params), dim=1)
                )
                scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
                nonanchor_hat = self.compress_nonanchor(
                    curr_y,
                    scales_nonanchor,
                    means_nonanchor,
                    symbols_list,
                    indexes_list,
                )
                uneven_groups[i] = anchor_hat + nonanchor_hat

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )
            y_string = encoder.flush()
            y_strings.append(y_string)

            y_strings_all.append(y_strings)
        return {
            "strings": [y_strings_all, z_strings],
            "shape": z.size()[-2:],
            "y_hat": uneven_groups,
        }

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        torch.cuda.synchronize()
        start_time = time.process_time()

        z_strings = strings[1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        y_hat_groups = []
        for i in range(5):
            y_strings = strings[0][i][0]
            decoder = RansDecoder()
            decoder.set_stream(y_strings)

            ctx_params_anchor = torch.zeros(
                [z_hat.size(0), 320 * 2, z_hat.size(2) * 4, z_hat.size(3) * 4],
                device=z_hat.device,
            )
            if i == 0:

                gaussian_params_anchor = self.entropy_parameters[i](
                    torch.cat([ctx_params_anchor, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
                anchor_hat = self.decompress_anchor(
                    scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets
                )

                ctx_params = self.context_prediction_models[i](anchor_hat)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat([ctx_params, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
                nonanchor_hat = self.decompress_nonanchor(
                    scales_nonanchor,
                    means_nonanchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )

                y_hat = anchor_hat + nonanchor_hat
                y_hat_groups.append(y_hat)
            else:
                channel_context = self.channel_context_models[i - 1](
                    torch.cat(y_hat_groups, dim=1)
                )
                gaussian_params_anchor = self.entropy_parameters[i](
                    torch.cat([ctx_params_anchor, channel_context, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
                anchor_hat = self.decompress_anchor(
                    scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets
                )

                ctx_params = self.context_prediction_models[i](anchor_hat)
                gaussian_params = self.entropy_parameters[i](
                    torch.cat([ctx_params, channel_context, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
                nonanchor_hat = self.decompress_nonanchor(
                    scales_nonanchor,
                    means_nonanchor,
                    decoder,
                    cdf,
                    cdf_lengths,
                    offsets,
                )

                y_hat = anchor_hat + nonanchor_hat
                y_hat_groups.append(y_hat)

        y_hat = torch.cat(y_hat_groups, dim=1)

        x_hat = self.g_s(y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time

        return {"x_hat": x_hat, "cost_time": cost_time, "y_hat": y_hat_groups}

    def ckbd_anchor_sequeeze(self, y):
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y):
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor):
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor):
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor

    def compress_anchor(
        self, anchor, scales_anchor, means_anchor, symbols_list, indexes_list
    ):
        # squeeze anchor to avoid non-anchor symbols
        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(
            anchor_squeeze, "symbols", means_anchor_squeeze
        )
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
        return anchor_hat

    def compress_nonanchor(
        self, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list
    ):
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(
            nonanchor_squeeze, "symbols", means_nonanchor_squeeze
        )
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(
            nonanchor_hat + means_nonanchor_squeeze
        )
        return nonanchor_hat

    def decompress_anchor(
        self, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets
    ):
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = decoder.decode_stream(
            indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        anchor_hat = (
            torch.Tensor(anchor_hat)
            .reshape(scales_anchor_squeeze.shape)
            .to(scales_anchor.device)
            + means_anchor_squeeze
        )
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(
        self, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets
    ):
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = decoder.decode_stream(
            indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        nonanchor_hat = (
            torch.Tensor(nonanchor_hat)
            .reshape(scales_nonanchor_squeeze.shape)
            .to(scales_nonanchor.device)
            + means_nonanchor_squeeze
        )
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat
