
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import os
import sys
import warnings

from natsort import natsorted
import glob

warnings.filterwarnings('ignore')


    
import argparse
import logging
import time

from torch.utils.data import DataLoader, RandomSampler

from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import prepare_frame, image_compress, load_model
from utils import get_scales, get_order_typ_list, update_buffer, select_references
from opt_helpers import get_best_down_ratio_prediction, get_best_down_ratio_compress

from config import *
from bd_rate import *


model_path = os.path.abspath('/home/akinyilmaz/icip2024/feature_domain_v12')
sys.path.insert(1, model_path)

from model import elic
from model import m

import matplotlib.pyplot as plt

logging.basicConfig(filename = log_name + ".log", level=logging.INFO)


def figure_staff(sequence, level, psnr_list, size_list, ft_list):
    fig, ax1 = plt.subplots() 
  
    ax1.set_xlabel('frame index') 
    ax1.set_ylabel('PSNR', color = 'red') 
    ax1.plot([i+1 for i in range(len(psnr_list))], psnr_list, color = 'red') 
    ax1.tick_params(axis ='y', labelcolor = 'red') 

    # Adding Twin Axes

    ax2 = ax1.twinx() 

    ax2.set_ylabel('bpp', color = 'blue') 
    ax2.plot([i+1 for i in range(len(size_list))], size_list, color = 'blue') 
    ax2.tick_params(axis ='y', labelcolor = 'blue') 
    
    plt.grid()
    plt.title(sequence+" level: "+str(level))
    plt.savefig("Figures/"+sequence+" level: "+str(level)+".png", format='png', facecolor="white", dpi=120, bbox_inches="tight")


def val_sequence_level(video_sequence, im_model, model, betas, device, order_list, typ_list, val_path, level): 
        
    h = 1080
    w = 1920

    beta = betas[level]

    psnr_list = [0 for i in range(len(video_sequence))]
    size_list = [0 for i in range(len(video_sequence))]
    loss_list = [0 for i in range(len(video_sequence))]
        

    buffer = []
    buffer_order = []

    with torch.no_grad():
        for order in order_list:
            typ = typ_list[order]
            if typ == "I":
                frame = prepare_frame(video_sequence[order], True).unsqueeze(0).to(device).float()
                dec, size = image_compress(frame, im_model, level)
                loss = torch.tensor([0])
            else:
                frame = prepare_frame(video_sequence[order], True).unsqueeze(0).to(device).float()
                ref1, ref2, order1, order2 = select_references(frame.detach().cpu(), order, buffer, buffer_order)

                scale1, scale2 = get_scales(order, order1, order2)
                
                down_ratio, best_pred_psnr = get_best_down_ratio_prediction(model, ref1.to(device), ref2.to(device), scale1, scale2, frame.to(device), level, beta)
                print(video_sequence[order], level, down_ratio, best_pred_psnr.item())
                
                dec_out = model(
                    xref1=ref1.to(device),
                    xref2=ref2.to(device),
                    xcur=frame,
                    scale1=scale1,
                    scale2=scale2,
                    s=level,
                    down_ratio=down_ratio,
                    )

                dec, rate, size = dec_out["x_hat"], dec_out["rate"], dec_out["size"]
                loss = beta * calculate_distortion_loss(dec, frame) + rate

            uint8_real = float_to_uint8(frame[0, :, :h, :w])
            uint8_dec_out = float_to_uint8(dec[0, :, :h, :w])

            psnr = PSNR(
                MSE(uint8_dec_out.type(torch.float), uint8_real.type(torch.float)),
                data_range=255
            )

            loss_list[order] = loss.item()
            psnr_list[order] = psnr.item()
            size_list[order] = (size.item())/(h*w)


            buffer, buffer_order = update_buffer(buffer, buffer_order, torch.clamp(dec.detach().cpu(), 0, 1), order)
            
                
    loss_list = [i for i in loss_list if i != "nan"]
    psnr_list = [i for i in psnr_list if i != "nan"]
    size_list = [i for i in size_list if i != "nan"]
        

    return loss_list, psnr_list, size_list
            

    
def validate_all(im_model, model, betas, device, val_path):
    
    all_levels_loss_list = []
    all_levels_psnr_list = []
    all_levels_size_list = []
    
    for level in range(levels):
        level_loss_list = []
        level_psnr_list = []
        level_size_list = []
        for sequence in folder_names:
            
            video_sequence = natsorted(glob.glob(val_path+sequence+"/*.png"))#[:17]

            
            order_list, typ_list = get_order_typ_list(intra_size, len(video_sequence))
                
            loss_list, psnr_list, size_list = val_sequence_level(video_sequence, 
                                                                 im_model, 
                                                                 model, betas, device, order_list, typ_list, val_path, level)
            
            level_loss_list.extend(loss_list)
            level_psnr_list.extend(psnr_list)
            level_size_list.extend(size_list)

            logging.info("level: "+str(level)+" -- sequence: "+sequence+" -- psnr: "+str(np.mean(psnr_list))+" -- bpp: "+str(np.mean(size_list)))
            logging.info("******************")
        
        avg_level_loss = np.mean(level_loss_list)
        avg_level_psnr = np.mean(level_psnr_list)
        avg_level_size = np.mean(level_size_list)
    
        all_levels_loss_list.append(avg_level_loss)
        all_levels_psnr_list.append(avg_level_psnr)
        all_levels_size_list.append(avg_level_size)
        
    return all_levels_loss_list, all_levels_psnr_list, all_levels_size_list


# In[ ]:


def main():
    
    
    i_model = []
    for i in range(levels):
        im = elic.ELIC()
        im.load_state_dict(torch.load(f"{intra_model_path}/ELIC_Rate_{i+4}.pth.tar")['state_dict'])
        im = im.to(device).float()
        for param in im.parameters():
            param.requires_grad = False
        im.eval()
        i_model.append(im)
    
    model = m.FlowGuidedB().to(device).float()
    
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage))
    
    
    model = model.eval()
    
    loss, psnr, bpp = validate_all(i_model, model, betas_mse, device, val_path)
    
    for level in range(levels):
        logging.info("level: " + str(level))
        logging.info("psnr: " + str(psnr[level]))
        logging.info("bpp: " + str(bpp[level]))
        logging.info("--------------------")
    


    
main()



