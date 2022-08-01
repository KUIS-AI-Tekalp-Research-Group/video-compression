
import argparse
import warnings
warnings.filterwarnings('ignore')

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from b_model.b_model import BidirFlowRef
from utils import load_model, PSNR, MSE

device = torch.device("cuda")

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref_1", type=str, default='../frames/img1.png')
parser.add_argument("--ref_2", type=str, default='../frames/img3.png')
parser.add_argument("--current", type=str, default='../frames/img2.png')
parser.add_argument("--bin", type=str, default='../frames/bits_B.bin')
parser.add_argument("--b_pretrained", type=str, default="../pretrained.pth")
parser.add_argument("--n", type=int, default=0, choices=[0, 1, 2, 3])
parser.add_argument("--l", type=float, default=1., choices=[0., 0.33, 0.66, 1.])
args = parser.parse_args()

# Build B model
model = BidirFlowRef(n=4, N=128).to(device).float()

if args.b_pretrained:
    checkpoint = torch.load(args.b_pretrained, map_location=device)
    model = load_model(model, checkpoint, exceptions=[])

model.flow_compressor.update(force=True)
model.residual_compressor.update(force=True)
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


def decode_B(model, x_before, x_after, string_flow, string_res, shape_flow, shape_res, n, l):
    x = torch.cat((x_before, x_after), 1)
    mv_before, mv_after, x_conc = model.process(x_before, x_after)

    flow_result = model.flow_compressor.decompress(string_flow, shape_flow, [n], l)

    flow_refinement = flow_result["x_hat"]
    mv_before_refined = mv_before + flow_refinement[:, :2, :, :]
    mv_after_refined = mv_after + flow_refinement[:, 2:4, :, :]
    
    x_b = model.backwarp(x_before, mv_before_refined)
    x_a = model.backwarp(x_after, mv_after_refined)
    
    temp = torch.cat((mv_before_refined,mv_after_refined,x,x_b,x_a),1)
    mask = F.sigmoid(model.Mask(temp))
    
    w1, w2 = 0.5*mask[:,0:1,:,:], 0.5*mask[:,1:2,:,:]
    x_comp = (w1*x_b+w2*x_a)/(w1+w2+1e-8)

    residual_result = model.residual_compressor.decompress(string_res, shape_res, [n], l)
    final = residual_result["x_hat"] + x_comp
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
    res_bits_dec = [[string0_res], [string1_res]]


with torch.no_grad():
    global gt, x_current, h, w

    ref1 = imageio.imread(args.ref_1).astype(float)
    ref2 = imageio.imread(args.ref_2).astype(float)
    (h, w, c) = ref1.shape
    x_before = process_frame(ref1)
    x_after = process_frame(ref2)

    gt = imageio.imread(args.current)
    x_current = process_frame(gt)

    print(f"Decode configuration:\n\tn: {args.n}, l: {args.l}")

    decoded_frame = decode_B(
        model, x_before, x_after,
        mv_bits_dec, res_bits_dec, 
        shape_mv, shape_res, 
        args.n, args.l
    )
    uint = float_to_uint8(decoded_frame[0].cpu().numpy())
    psnr = PSNR(
        MSE(uint[:h, :w].astype(np.float64), gt.astype(np.float64)), 
        data_range=255
    )
    size = os.stat("../frames/bits_B.bin").st_size
    print(f"PSNR: {psnr:.4f}")
    print(f"Size: {size} bytes")
    imageio.imwrite("../frames/decoded.png", uint[:h, :w])