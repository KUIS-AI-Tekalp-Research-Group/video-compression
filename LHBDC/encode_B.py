#!/usr/bin/env python

import os
import math
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from compressai.zoo import mbt2018_mean

from model import m

device = torch.device("cuda")


parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref_1", default='frames/ref_1.png')
parser.add_argument("--ref_2", default='frames/ref_2.png')
parser.add_argument("--current", default='frames/current.png')
parser.add_argument("--bin", default='bits_B.bin')
parser.add_argument("--l", type=int, default=436, choices=[228, 436])
args = parser.parse_args()



model = m.Model()
model.load_state_dict(torch.load(f"pretrained_weights/compression_{args.l}.pth", map_location=lambda storage, loc: storage)["state_dict"])
model.mv_compressor.update(force=True)
model.residual_compressor.update(force=True)
model = model.to(device).float()
model = model.eval()


def normalize(tensor):
    norm = (tensor)/255.
    return norm

def float_to_uint8(image):
    clip = np.clip(image,0,1)*255.
    im_uint8 = np.round(clip).astype(np.uint8).transpose(1,2,0)
    return im_uint8

def pad(im):
    (m,c,w,h) = im.size()

    p1 = (64 - (w % 64)) % 64
    p2 = (64 - (h % 64)) % 64
    
    pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
    return pad(im)

def process_frame(img):
    x = img.transpose(2,0,1)
    (c,h,w) = x.shape
    x = x.reshape(1,c,h,w)
    x = normalize(torch.from_numpy(x).to(device).float())
    x  = pad(x)
    return x

def ups(flow):
    upsample_flow = nn.Upsample(scale_factor=4, mode='bilinear')
    return upsample_flow(flow)


def encode_B(model, x_after, x_current, x_before):
        

    flow_ba = F.avg_pool2d(model.FlowNet(x_before, x_after) / 2., 4)
    flow_ab = F.avg_pool2d(model.FlowNet(x_after, x_before) / 2., 4)
    nnn,ccc,hhh,www = flow_ab.size()

    flow_ba = pad(flow_ab)
    flow_ab = pad(flow_ba)

    flow_cb = F.avg_pool2d(model.FlowNet(x_current, x_before), 4)
    flow_ca = F.avg_pool2d(model.FlowNet(x_current, x_after), 4)

    flow_cb = pad(flow_cb)
    flow_ca = pad(flow_ca)

    diff_flow = torch.cat([(flow_cb - flow_ab), (flow_ca - flow_ba)], dim=1) # 850 ms

    flow_result = model.mv_compressor(diff_flow)
    flow_cb_hat, flow_ca_hat = torch.chunk(flow_result["x_hat"], 2, dim=1)
    flow_cb_hat = flow_cb_hat + flow_ab
    flow_cb_hat = model.upsample_flow(flow_cb_hat[:, :, :hhh, :www])
    flow_ca_hat = flow_ca_hat + flow_ba
    flow_ca_hat = model.upsample_flow(flow_ca_hat[:, :, :hhh, :www])

    mv_bits = model.mv_compressor.compress(diff_flow) # 25 ms

    fw, bw = model.backwarp(x_before, flow_cb_hat), model.backwarp(x_after, flow_ca_hat)
    mask = model.masknet(torch.cat([fw, bw], dim=1)).repeat([1, 3, 1, 1]) # 130 ms

    x_current_hat = mask*fw + (1.0 - mask)*bw

    res = x_current - x_current_hat
    res_bits = model.residual_compressor.compress(res)
    return mv_bits, res_bits


with torch.no_grad():
    x_before = process_frame(imageio.imread(args.ref_1).astype(float))
    x_after = process_frame(imageio.imread(args.ref_2).astype(float))
    x_current = process_frame((imageio.imread(args.current).astype(float)))
    mv_bits, res_bits = encode_B(model, x_after, x_current, x_before)

with open(args.bin, "wb") as ff:
    ff.write(np.array(args.l, dtype=np.uint32).tobytes())
    ff.write(np.array(mv_bits["shape"], dtype=np.uint16).tobytes())
    ff.write(np.array(len(mv_bits["strings"][0][0]), dtype=np.uint32).tobytes())
    ff.write(np.array(len(mv_bits["strings"][1][0]), dtype=np.uint32).tobytes())
    
    ff.write(np.array(res_bits["shape"], dtype=np.uint16).tobytes())
    ff.write(np.array(len(res_bits["strings"][0][0]), dtype=np.uint32).tobytes())
    
    ff.write(mv_bits["strings"][0][0])
    ff.write(mv_bits["strings"][1][0])
    ff.write(res_bits["strings"][0][0])
    ff.write(res_bits["strings"][1][0])

