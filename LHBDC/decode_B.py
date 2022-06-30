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

device = torch.device("cuda")


parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref_1", default='frames/ref_1.png')
parser.add_argument("--ref_2", default='frames/ref_2.png')
parser.add_argument("--bin", default='bits_B.bin')
args = parser.parse_args()

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



def decode_B(x_before, x_after, model, string_flow, string_res, shape_flow, shape_res):
    
    flow_ba = F.avg_pool2d(model.FlowNet(x_before, x_after) / 2., 4) 
    flow_ab = F.avg_pool2d(model.FlowNet(x_after, x_before) / 2., 4)
    nnn,ccc,hhh,www = flow_ab.size()

    flow_ba = pad(flow_ab)
    flow_ab = pad(flow_ba) 

    flow_result = model.mv_compressor.decompress(string_flow, shape_flow)["x_hat"] 
    flow_cb_hat, flow_ca_hat = torch.chunk(flow_result, 2, dim=1)
    flow_cb_hat = flow_cb_hat + flow_ab
    flow_cb_hat = model.upsample_flow(flow_cb_hat[:, :, :hhh, :www])
    flow_ca_hat = flow_ca_hat + flow_ba
    flow_ca_hat = model.upsample_flow(flow_ca_hat[:, :, :hhh, :www])

    fw, bw = model.backwarp(x_before, flow_cb_hat), model.backwarp(x_after, flow_ca_hat)
    mask = model.masknet(torch.cat([fw, bw], dim=1)).repeat([1, 3, 1, 1]) 
    x_current_hat = mask*fw + (1.0 - mask)*bw

    res_result = model.residual_compressor.decompress(string_res, shape_res)["x_hat"] 
    final = res_result + x_current_hat

    return final

with open(args.bin, "rb") as ff:
    l = np.frombuffer(ff.read(4), dtype=np.uint32)[0]
    shape_mv = torch.Size(np.frombuffer(ff.read(4), dtype=np.uint16).astype(np.int))
    len0_mv = np.frombuffer(ff.read(4), dtype=np.uint32)
    len1_mv = np.frombuffer(ff.read(4), dtype=np.uint32)
    
    shape_res = torch.Size(np.frombuffer(ff.read(4), dtype=np.uint16).astype(np.int))
    len0_res = np.frombuffer(ff.read(4), dtype=np.uint32)
    
    string0_mv = ff.read(np.int(len0_mv))
    string1_mv = ff.read(np.int(len1_mv))
    
    string0_res = ff.read(np.int(len0_res))
    string1_res = ff.read()
    
    mv_bits_dec = [[string0_mv], [string1_mv]]
    res_bits_dec = [[string0_mv], [string1_mv]]




model = m.Model()
model.load_state_dict(torch.load(f"pretrained_weights/compression_{l}.pth", map_location=lambda storage, loc: storage)["state_dict"])
model.mv_compressor.update(force=True)
model.residual_compressor.update(force=True)
model = model.to(device).float()
model = model.eval()

with torch.no_grad():
    x_before = process_frame(imageio.imread(args.ref_1).astype(float))
    x_after = process_frame(imageio.imread(args.ref_2).astype(float))
    decoded_frame = decode_B(x_before, x_after, model, mv_bits_dec, res_bits_dec, shape_mv, shape_res)


