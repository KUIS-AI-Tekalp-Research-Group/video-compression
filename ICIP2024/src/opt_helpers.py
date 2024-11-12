#!/usr/bin/env python
# coding: utf-8

# # 2. Utils

# In[ ]:


import torch
from torch import optim
import torchvision
import numpy as np
from natsort import natsorted
import glob
import random
import sys
import math
import torch.nn as nn
import torch.nn.functional as F

import logging

def prediction_flowonly(model, xcur, xref1, xref2, scale1, scale2, down_ratio):
    scale1, scale2 = model.convert_scales(scale1, scale2, xref1)
    
    flow_2_to_1_cur, flow_1_to_2_cur = model.estimate_flow(xref1, xref2, down_ratio).chunk(2, 1)
    flow_2_to_1_cur = F.interpolate(flow_2_to_1_cur, scale_factor=2, mode="bilinear", align_corners=False)*2
    flow_1_to_2_cur = F.interpolate(flow_1_to_2_cur, scale_factor=2, mode="bilinear", align_corners=False)*2
    flow_2_to_1_cur = flow_2_to_1_cur * scale1
    flow_1_to_2_cur = flow_1_to_2_cur * scale2
            
    wref1 = model.warp(xref1, flow_2_to_1_cur)
    wref2 = model.warp(xref2, flow_1_to_2_cur)
    
    mask = 0.5
    xcomp = mask*wref1 + (1-mask)*wref2

    return xcomp


def get_best_down_ratio_prediction(model, xref1, xref2, scale1, scale2, xcur, level, beta):
    best_pred_psnr = 0
    for down_ratio in [1, 2, 4, 8, 16]:
        x_hat = prediction_flowonly(model, xcur, xref1, xref2, scale1, scale2, down_ratio)
        psnr = PSNR(MSE(torch.clamp(x_hat, 0, 1), xcur), 1)
        if psnr > best_pred_psnr:
            best_pred_psnr = psnr
            best_down_ratio = down_ratio

        
    return best_down_ratio, best_pred_psnr

def get_best_down_ratio_compress(model, xref1, xref2, scale1, scale2, xcur, level, beta):
    h = 1080
    w = 1920
    best_pred_loss = 1e6
    for down_ratio in [1, 2, 4, 8, 16]:
        dec_out = model(
                    xref1=xref1,
                    xref2=xref2,
                    xcur=xcur,
                    scale1=scale1,
                    scale2=scale2,
                    s=level,
                    down_ratio=down_ratio,
                    )

        dec, rate, size = dec_out["x_hat"], dec_out["rate"], dec_out["size"]
        loss = beta * calculate_distortion_loss(dec, xcur) + rate
        if loss < best_pred_loss:
            best_pred_loss = loss
            best_down_ratio = down_ratio


    return best_down_ratio, best_pred_loss


def adjust_level(model, xref1, xref2, scale1, scale2, xcur, icoded_xcur, level, max_levels, maxdiff, mindiff, leveldiff):
    out = prediction_flowonly(model, xref1, xref2, scale1, scale2)
    out2 = prediction_offset(model, xref1, xref2, scale1, scale2)

    psnrref = PSNR(MSE(torch.clamp(icoded_xcur, 0, 1), xcur), 1).item()
    psnrcur = PSNR(MSE(torch.clamp(out, 0, 1), xcur), 1).item()
    psnrcur2 = PSNR(MSE(torch.clamp(out2, 0, 1), xcur), 1).item()



    
    psnrdiff = psnrref - psnrcur
    psnrdiff_ = min(max(0, psnrdiff), maxdiff)
    
    I_compress = False
    No_compress = False
    if psnrdiff_ > maxdiff:
        I_compress = False
    if psnrdiff_ < mindiff:
        No_compress = False

    qualratio = psnrdiff_/maxdiff

    return I_compress, No_compress, min(max((level+leveldiff)*qualratio, 0), max_levels)

    
    
def select_references(order, buffer, buffer_order):
    k = 2
    if len(buffer) == 1:
        k = 1
    
    d = torch.from_numpy(np.array([abs(i-order) for i in buffer_order]))  
    ind = list(torch.topk(d, k, largest=False).indices.numpy())    
    
    if k == 1:
        ref1 = buffer[ind[0]]
        ref2 = buffer[ind[0]]
        order1 = buffer_order[ind[0]]
        order2 = buffer_order[ind[0]]
        
    else:
        min_ind, max_ind = ind[1], ind[0]
        if buffer_order[ind[0]] < buffer_order[ind[1]]:
            min_ind, max_ind = ind[0], ind[1]
        ref1 = buffer[min_ind]
        ref2 = buffer[max_ind]
        order1 = buffer_order[min_ind]
        order2 = buffer_order[max_ind]
        
    return ref1, ref2, order1, order2



def get_scales(order, order1, order2):
    o2o1 = order2 - order1
    o1o2 = order1 - order2
    
    oo1 = order - order1
    oo2 = order - order2
    
    if torch.is_tensor(order) == False:
        if o2o1 == 0:
            scale1 = 0
            scale2 = 0
        else:
            scale1 = oo1 / o2o1
            scale2 = oo2 / o1o2

    else:
        scale1 = oo1 / o2o1
        scale2 = oo2 / o1o2


    return scale1, scale2

def normalize(tensor):
    norm = (tensor) / 255.
    return norm


# In[2]:


def float_to_uint8(image):
    clip = torch.clamp(image, 0., 1.) * 255.
    im_uint8 = torch.round(clip).type(torch.uint8)
    return im_uint8


# In[3]:


def MSE(gt, pred):
    mse = torch.mean((gt - pred) ** 2)
    return mse


# In[4]:


def PSNR(mse, data_range):
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr


# In[5]:


def calculate_distortion_loss(out, real):
    """Mean Squared Error"""
    distortion_loss = torch.mean((out - real) ** 2)
    return distortion_loss


# In[6]:


def pad(im):
    """Padding to fix size at validation"""
    (b, c, w, h) = im.size()

    p1 = (64 - (w % 64)) % 64
    p2 = (64 - (h % 64)) % 64

    pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
    return pad(im).squeeze(0)
        
