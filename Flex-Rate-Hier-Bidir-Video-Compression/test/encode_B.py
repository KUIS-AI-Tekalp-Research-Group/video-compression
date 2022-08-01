
import os, sys
import argparse
import warnings
warnings.filterwarnings('ignore')

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio

from b_model.b_model import BidirFlowRef
from utils import load_model, MSE, PSNR

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


def encode_B(model, x_before, x_current, x_after, n=None, l=1., train=False):
    """
    encode flow and residual information
    x_before: past frame,
    x_current: frame to be encoded,
    x_after: future frame,
    n: compression quality (choosing gain vector),
    l: compression quality exponential interpolation factor
    """
    x = torch.cat((x_before, x_after), 1)

    mv_before, mv_after, x_conc = model.process(x_before, x_after)
    x_input = torch.cat((x_conc, x_current), 1)

    mv_bits = model.flow_compressor.compress(
        x_input, [n], l
    )
    
    flow_result = model.flow_compressor(x_input, [n], l, train)
    flow_hat = flow_result["x_hat"]

    mv_before_refined = mv_before + flow_hat[:, :2, :, :]
    mv_after_refined = mv_after + flow_hat[:, 2:4, :, :]
    
    x_b = model.backwarp(x_before, mv_before_refined)
    x_a = model.backwarp(x_after, mv_after_refined)
    
    temp = torch.cat((mv_before_refined,mv_after_refined,x,x_b,x_a),1)
    mask = F.sigmoid(model.Mask(temp))
    
    w1, w2 = 0.5*mask[:,0:1,:,:], 0.5*mask[:,1:2,:,:]
    x_comp = (w1*x_b+w2*x_a)/(w1+w2+1e-8)

    residual = x_current - x_comp
    res_bits = model.residual_compressor.compress(residual, [n], l)
    return mv_bits, res_bits


with torch.no_grad():
    x_before = process_frame(imageio.imread(args.ref_1).astype(float))
    x_after = process_frame(imageio.imread(args.ref_2).astype(float))
    x_current = process_frame((imageio.imread(args.current).astype(float)))

    print(f"Encode configuration:\n\tn: {args.n}, l: {args.l}")

    mv_bits, res_bits = encode_B(
        model, x_before, x_current, x_after,
        n=args.n, l=args.l
    )

with open(args.bin, "wb") as ff:
    # interpolation factor
    ff.write(np.array(args.l, dtype=np.uint32).tobytes())
    # shape of tensor z for optical flow refinement
    ff.write(np.array(mv_bits["shape"], dtype=np.uint16).tobytes())
    # length of y tensor for opical flow refinement
    ff.write(np.array(len(mv_bits["strings"][0][0]), dtype=np.uint32).tobytes())
    # length of z tensor for opical flow refinement
    ff.write(np.array(len(mv_bits["strings"][1][0]), dtype=np.uint32).tobytes())

    # shape of tensor z for frame residual
    ff.write(np.array(res_bits["shape"], dtype=np.uint16).tobytes())
    # length of y tensor for frame residual
    ff.write(np.array(len(res_bits["strings"][0][0]), dtype=np.uint32).tobytes())
    # length of y tensor for frame residual
    # ff.write(np.array(len(res_bits["strings"][1][0]), dtype=np.uint32).tobytes())
    
    # information about the optical flow refinement and frame residual
    ff.write(mv_bits["strings"][0][0])
    ff.write(mv_bits["strings"][1][0])
    ff.write(res_bits["strings"][0][0])
    ff.write(res_bits["strings"][1][0])
