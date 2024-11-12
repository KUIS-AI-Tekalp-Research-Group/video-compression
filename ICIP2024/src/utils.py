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

def get_best_down_ratio(model, xref1, xref2, scale1, scale2, xcur, level):
    best_rate = 1000000
    for down_ratio in [1, 2, 4, 8]:
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

        
        if rate.item() < best_rate:
            best_rate = rate.item()
            best_down_ratio = down_ratio
            
    return best_down_ratio


def get_best_down_ratio_prediction(model, xref1, xref2, scale1, scale2, xcur, level):
    best_rate = 1000000
    for down_ratio in [1, 2, 4, 8, 16, 32]:
        x_hat = prediction_flowonly(model, xref1, xref2, scale1, scale2, down_ratio)
        
        
            
    return best_down_ratio

def prediction_offset(model, xref1, xref2, scale1, scale2):
    scale1, scale2 = model.convert_scales(scale1, scale2)
    flow_l3 = model.estimate_flow(xref1, xref2, 1)
    fref1, fref2, _ = model.get_ms_features(xref1, xref2, xref2)

    flow_cur1_l3, flow_cur2_l3, wref1_l3, wref2_l3, flow_l2 = model.get_warpedrefs_at_layer(
                                                                                            fref1[2], fref2[2],
                                                                                            flow_l3, scale1, scale2
                                                                                            )

    flow_cur1_l2, flow_cur2_l2, wref1_l2, wref2_l2, flow_l1 = model.get_warpedrefs_at_layer(
                                                                                            fref1[1], fref2[1],
                                                                                            flow_l2, scale1, scale2
                                                                                            )
    
    flow_cur1_l1, flow_cur2_l1, wref1_l1, wref2_l1, _ = model.get_warpedrefs_at_layer(
                                                                                            fref1[0], fref2[0],
                                                                                            flow_l1, scale1, scale2
                                                                                            )


    o1 = torch.zeros((flow_cur1_l3.shape[0], 27*8, flow_cur1_l3.shape[2], flow_cur1_l3.shape[3]))
    o2 = torch.zeros((flow_cur1_l3.shape[0], 27*8, flow_cur1_l3.shape[2], flow_cur1_l3.shape[3]))
    o1[:,18*8:] = 0
    o2[:,18*8:] = 0

    offset_result = {}
    offset_result['offset3'] = torch.cat([o1, o2], dim=1).to(flow_cur1_l2.device).float()

    o1 = torch.zeros((flow_cur1_l2.shape[0], 27*8, flow_cur1_l2.shape[2], flow_cur1_l2.shape[3]))
    o2 = torch.zeros((flow_cur1_l2.shape[0], 27*8, flow_cur1_l2.shape[2], flow_cur1_l2.shape[3]))
    o1[:,18*8:] = 0
    o2[:,18*8:] = 0

    offset_result['offset2'] = torch.cat([o1, o2], dim=1).to(flow_cur1_l2.device).float()

    o1 = torch.zeros((flow_cur1_l1.shape[0], 27*8, flow_cur1_l1.shape[2], flow_cur1_l1.shape[3]))
    o2 = torch.zeros((flow_cur1_l1.shape[0], 27*8, flow_cur1_l1.shape[2], flow_cur1_l1.shape[3]))
    o1[:,18*8:] = 0
    o2[:,18*8:] = 0

    offset_result['offset1'] = torch.cat([o1, o2], dim=1).to(flow_cur1_l2.device).float()


    x_comp_l3 = model.get_deformed_maps(offset_result, flow_cur1_l3, flow_cur2_l3, fref1[2], fref2[2], model.deconv_l3_1, model.deconv_l3_2, 3)
    x_comp_l2 = model.get_deformed_maps(offset_result, flow_cur1_l2, flow_cur2_l2, fref1[1], fref2[1], model.deconv_l2_1, model.deconv_l2_2, 2)
    x_comp_l1 = model.get_deformed_maps(offset_result, flow_cur1_l1, flow_cur2_l1, fref1[0], fref2[0], model.deconv_l1_1, model.deconv_l1_2, 1)

    x_hat = model.reconstructor(x_comp_l1, x_comp_l2, x_comp_l3)
    return x_hat

def prediction_flowonly(model, xref1, xref2, scale1, scale2, down_ratio):
    scale1, scale2 = model.convert_scales(scale1, scale2)

    xref1_down = F.avg_pool2d(xref1, down_ratio)
    xref2_down = F.avg_pool2d(xref2, down_ratio)

    flow_1_to_2_cur = model.estimate_flow(xref1_down, xref2_down)*scale2
    flow_2_to_1_cur = model.estimate_flow(xref2_down, xref1_down)*scale1
    
    flow_1_to_2_cur = F.interpolate(flow_1_to_2_cur, scale_factor=down_ratio, mode="bilinear", align_corners=False)*down_ratio
    flow_2_to_1_cur = F.interpolate(flow_2_to_1_cur, scale_factor=down_ratio, mode="bilinear", align_corners=False)*down_ratio
        
    wref1 = model.warp(xref1, flow_2_to_1_cur)
    wref2 = model.warp(xref2, flow_1_to_2_cur)
    
    x_hat = (wref1 + wref2)*0.5
    return x_hat


def adjust_level(model, xref1, xref2, scale1, scale2, xcur, icoded_xcur, level, max_levels, maxdiff, mindiff, leveldiff):
    out = prediction_flowonly(model, xref1, xref2, scale1, scale2)
    out2 = prediction_offset(model, xref1, xref2, scale1, scale2)

    psnrref = PSNR(MSE(torch.clamp(icoded_xcur, 0, 1), xcur), 1).item()
    psnrcur = PSNR(MSE(torch.clamp(out, 0, 1), xcur), 1).item()
    psnrcur2 = PSNR(MSE(torch.clamp(out2, 0, 1), xcur), 1).item()


    print(psnrref, psnrcur, psnrcur2)
    print('------')
    
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

    
    
def select_references(xcur, order, buffer, buffer_order):
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


def update_buffer(buffer, buffer_order, new_frame, order, l=32):
    updated_buffer = buffer + [new_frame]
    updated_buffer_order = buffer_order + [order]
    if len(buffer) < l:
        return updated_buffer, updated_buffer_order
    else:
        return updated_buffer[1:], updated_buffer_order[1:]

def get_order_typ_list(intra_size, frame_number):
    order = [16, 8, 4, 12, 2, 14, 6, 10, 1, 15, 3, 13, 5, 11, 7, 9]

    o = [0]

    lll = len(order)

    ff = (frame_number - 1)%intra_size

    for i in range(0, frame_number-1):
        o.append(order[i%lll] + (i//lll)*lll)

    if ff != 0:
        m = max(o[:-ff])
        o[-ff:] = [(m + ff - i) for i in range(ff)]
    
    l = ["B" for i in range(frame_number)]
    l = ["I" if i % intra_size == 0 else l[i] for i in range(frame_number)]
    l[-1] = "I"

    indices = [i for i, x in enumerate(l) if x == "I"]
    """if frame_number == 300:
        o[-11:] = [299, 293, 290, 296, 289, 291, 292, 294, 295, 297, 298]
    if frame_number == 600:
        o[-23:] = [599, 588, 582, 594, 579, 585, 591, 597, 577, 578, 580, 581, 583, 584, 586, 587, 589, 590, 592, 593, 595, 596, 598]"""
    
    if frame_number == 300:
        o[-11:] = [299, 293, 290, 296, 289, 291, 292, 294, 295, 297, 298]
    if frame_number == 600:
        o[-7:] = [599, 595, 593, 597, 594, 596, 598]
    
        
    return o, l


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

# In[1]:
def prepare_frame(frame, p):
    im = torchvision.io.read_image(frame)
    im = normalize(im).unsqueeze(0)
    if p:
        im = pad(im)
    else:
        im = im.squeeze(0)
    return im


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
        



def image_compress(im, compressors, n):
    compressor = compressors[n]
    compressor = compressor.eval()
    out = compressor(im)
    dec = out["x_hat"]
    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out["likelihoods"].values()
    )

    return dec, size_image




def load_model(model, pretrained_dict, exceptions):
    """
    Load the model parameters from a dictionary. The dictionary must have key names same
    as the model attributes (which are submodules). The save_model() function is designed
    to be matching with this function.
    """
    
    model_child_names = [name for name, _ in model.named_children()]
    
    for name, submodule in pretrained_dict.items():
        # If we don't want to load a module, we skip it
        if name in exceptions:
            continue
            
        if name in model_child_names:
            message = getattr(model, name).load_state_dict(submodule)
            logging.info(name + ": " + str(message))
    return model

