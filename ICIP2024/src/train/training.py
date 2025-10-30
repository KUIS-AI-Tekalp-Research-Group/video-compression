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

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from model import elic
from model import m

from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import VimeoTrainDataset
from utils import prepare_frame, image_compress
from utils import configure_seeds, configure_optimizers
from utils import get_scales, get_order_typ_list, update_buffer, select_references

from config import *
from bd_rate import *

import copy
# In[ ]:


logging.basicConfig(filename = log_name + ".log", level=logging.INFO)


def train_one_step(train_batch, model, im_model, optimizer, aux_optimizer, betas, device, rng, iteration):
        
    x1 = train_batch[:, 0:3].to(device).float()
    x2 = train_batch[:, 3:6].to(device).float()
    x3 = train_batch[:, 6:9].to(device).float()
    x4 = train_batch[:, 9:12].to(device).float()
    x5 = train_batch[:, 12:15].to(device).float()

    level = rng.choices(population=[i for i in range(levels)],k=1)[0]
    
    beta = betas[level]
    
    down_ratio = rng.choices(population=[1, 2, 4],k=1)[0]

    with torch.no_grad():
        dec1, _ = image_compress(x1, im_model, level)
        dec5, _ = image_compress(x5, im_model, level)

    output3 = model(
            xref1=dec1, 
            xref2=dec5,
            xcur=x3,
            scale1=0.5,
            scale2=0.5,
            s=level,
            down_ratio=down_ratio,
        )

    dist_loss3 = beta * calculate_distortion_loss(output3["x_hat"], x3)
    rate_loss3 = output3["rate"] 


    if iteration < stage1_train_step:
        loss = dist_loss3 + rate_loss3
    else:
        output2 = model(
                xref1=dec1, 
                xref2=output3["x_hat"],
                xcur=x2,
                scale1=0.5,
                scale2=0.5,
                s=level,
                down_ratio=down_ratio,
            )
    
        output4 = model(
                xref1=output3["x_hat"], 
                xref2=dec5,
                xcur=x4,
                scale1=0.5,
                scale2=0.5,
                s=level,
                down_ratio=down_ratio,
            )


        dist_loss2 = beta * calculate_distortion_loss(output2["x_hat"], x2)
        rate_loss2 = output2["rate"] 
        dist_loss4 = beta * calculate_distortion_loss(output4["x_hat"], x4)
        rate_loss4 = output4["rate"] 

        dist_loss = (dist_loss3 + dist_loss2 + dist_loss4)/3.
        rate_loss = (rate_loss3 + rate_loss2 + rate_loss4)/3.
      
        loss = dist_loss + rate_loss
    
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    aux_loss = (model.offset_compressor.aux_loss() + model.residual_compressor.aux_loss())*0.5
    aux_loss.backward()
    aux_optimizer.step()
        
    return loss.item(), aux_loss.item()


# In[ ]:


def val_sequence_level(sequence, im_model, model, betas, device, order_list, typ_list, val_path, level): 
    down_ratio = 1
    
    h = 1080
    w = 1920

    beta = betas[level]

    loss_list = ["nan" for i in range(total_val_frames)]
    psnr_list = ["nan" for i in range(total_val_frames)]
    size_list = ["nan" for i in range(total_val_frames)]

    buffer = []
    buffer_order = []

    video_sequence = natsorted(glob.glob(val_path+sequence+"/*.png"))[:total_val_frames]
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
            
            order_list, typ_list = get_order_typ_list(intra_size, total_val_frames)
                
            loss_list, psnr_list, size_list = val_sequence_level(sequence, 
                                                                 im_model, 
                                                                 model, betas, device, order_list, typ_list, val_path, level)
            
            level_loss_list.extend(loss_list)
            level_psnr_list.extend(psnr_list)
            level_size_list.extend(size_list)
        
        avg_level_loss = np.mean(level_loss_list)
        avg_level_psnr = np.mean(level_psnr_list)
        avg_level_size = np.mean(level_size_list)
    
        all_levels_loss_list.append(avg_level_loss)
        all_levels_psnr_list.append(avg_level_psnr)
        all_levels_size_list.append(avg_level_size)
        
    return all_levels_loss_list, all_levels_psnr_list, all_levels_size_list


# In[ ]:


def main():
    rng = configure_seeds(random_seed, torch_seed)
    
    i_model = []
    for i in range(levels):
        im = elic.ELIC()
        im.load_state_dict(torch.load(f"{intra_model_path}/ELIC_Rate_{i+4}.pth.tar", map_location=lambda storage, loc: storage)['state_dict'])
        im = im.to(device).float()
        for param in im.parameters():
            param.requires_grad = False
        im.eval()
        i_model.append(im)
    
    
    model = m.FlowGuidedB().to(device).float()
    
    
    args = (learning_rate, aux_learning_rate, patience, min_lr)
    optimizer, aux_optimizer, _ = configure_optimizers(model, args)

    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))
    if pretrained_optimizer:
        optimizer.load_state_dict(torch.load(pretrained_optimizer, map_location=lambda storage, loc: storage))
    if pretrained_aux_optimizer:
        aux_optimizer.load_state_dict(torch.load(pretrained_aux_optimizer, map_location=lambda storage, loc: storage))
        

    for g in optimizer.param_groups:
        g['lr'] = learning_rate
    for g in aux_optimizer.param_groups:
        g['lr'] = aux_learning_rate
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    
    logging.info("Num. params: " + str(params))
    
    
    
    train_dataset = VimeoTrainDataset(
                        train_path, 
                        patch_size,
                        rng=rng,
                        dtype="png"
                    )    
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, drop_last=True, shuffle=True, pin_memory=True)
    generator = iter(train_loader)

    model.train()
    
    iteration = 1
    average_train_loss = 0

    best_loss = 10000000
    best_psnr = 0
    best_bpp = 0
    best_bd = 10000000
    
    for iteration in range(1, total_train_step+1):
        try:
            train_batch = next(generator)
            
        except StopIteration:
            generator = iter(train_loader)
            train_batch = next(generator)
         
        loss, aux_loss = train_one_step(
            train_batch=train_batch, 
            model=model,
            im_model=i_model,
            optimizer=optimizer, 
            aux_optimizer=aux_optimizer,
            betas=betas_mse,
            device=device,
            rng=rng,
            iteration=iteration,
        )
            
        print(iteration, loss, aux_loss)
        
        average_train_loss += loss

        
        if iteration % train_step == 0:
            logging.info("iteration: " + str(iteration))
            logging.info("lr: " + str(optimizer.param_groups[0]["lr"]))
            logging.info("train loss: " + str(average_train_loss/train_step))
            
            
            model.eval()
            loss, psnr, bpp = validate_all(i_model, model, betas_mse, device, val_path)
            
            bd = BD_RATE(ref_bpp, ref_psnr, bpp, psnr)

            avg_loss = np.mean(loss)
            avg_psnr = np.mean(psnr)
            avg_bpp = np.mean(bpp)            
            
            if bd < best_bd:
                
                logging.info("NEW BEST !!!!")
                
                best_bd = bd
                best_loss = avg_loss
                best_psnr = avg_psnr
                best_bpp = avg_bpp

                torch.save(model.state_dict(), model_save_dir) 
                torch.save(optimizer.state_dict(), optimizer_save_dir) 
                torch.save(aux_optimizer.state_dict(), aux_optimizer_save_dir) 
                
            for level in range(levels):
                logging.info("level: " + str(level))
                logging.info("loss: " + str(loss[level]))
                logging.info("psnr: " + str(psnr[level]))
                logging.info("bpp: " + str(bpp[level]))
                logging.info("--------------------")
            
        
            logging.info("avg level loss: " + str(avg_loss))
            logging.info("avg level psnr: " + str(avg_psnr))
            logging.info("avg level bpp: " + str(avg_bpp))
            logging.info("avg level bd: " + str(bd))
            logging.info("********************")
            logging.info("avg level psnr at best: " + str(best_psnr))
            logging.info("avg level bpp at best: " + str(best_bpp))
            logging.info("avg level bd at best: " + str(best_bd))
            logging.info("avg level loss at best: " + str(best_loss))
            logging.info("/////////////////////////")                
        
        
            average_train_loss = 0
            model.train()
        
        if iteration == change_lr_step:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1.e-5
            for param_group in aux_optimizer.param_groups:
                param_group['lr'] = 1.e-4

        if iteration >= total_train_step + 1:
            break

main()


