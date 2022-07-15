#!/usr/bin/env python
# coding: utf-8


# # 1. Test code


import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
import argparse
import logging
import math

import matplotlib.pyplot as plt
from compressai.zoo import mbt2018_mean
from torch.utils.data import DataLoader

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from model.m import Model

from utils import (MSE, PSNR, TestInfographic, UVGTestDataset, float_to_uint8)

torch.backends.cudnn.benchmark = True


# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the test
parser.add_argument("--project_name", type=str, default="LHBDC_test")                        # Project name
parser.add_argument("--model_name", type=str, default="Single_level_1626")                      # Model name
          
parser.add_argument("--test_path", type=str, default="/datasets/UVG/full_test/")                 # Dataset paths

parser.add_argument("--test_gop_size", type=int, default=8)                                    # Test gop sizes
parser.add_argument("--i_interval", type=int, default=8)                                       # Test i compression interval
parser.add_argument("--test_skip_frames", type=int, default=1)                                  # Test skip frames
parser.add_argument("--test_numbers", type=int, default=None)                                      # How many times to test on a video

parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--b_pretrained", type=str, default="../new_compression_1626.pth")                # Load model from this file
parser.add_argument("--i_qual", type=int, default=7)
parser.add_argument("--lmbda", type=int, default=1626)
# parser.add_argument("--p_pretrained", type=str, default="../SSF_gained.pth")                  # Load model from this file

parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.save_name = args.model_name

logging.basicConfig(filename= args.save_name + "_test.log", level=logging.INFO)

args.i_interval /= args.test_gop_size

device = torch.device(args.device)
    
# Frame order for decoding
coding_order = [0, 8, 4, 2, 1, 3, 6, 5, 7]
# prev_frame, future_frame, frame_level
decoding_info = {4: [0, 8], 2: [0, 4], 1: [0, 2], 3: [2, 4], 6: [4, 8], 5: [4, 6], 7: [6, 8]}
                 
hier_levels = {4: 0, 2: 1, 1: 2, 3: 2, 6: 1, 5: 2, 7: 2}

# ### Test Function

def image_compress(im, compressor):
    out = compressor(im)
    dec = out["x_hat"]
    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out["likelihoods"].values()
    )

    return dec, size_image


def test(b_model, i_model, device, args):
    """
    test_loader: Test loader for UVG
    model: B-frame compressor model
    device: cuda or cpu
    """
    
    with torch.no_grad():
        coding_order_eff = coding_order[2:]
        
        folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]
        
        infographic = TestInfographic(args.i_qual, folder_names)        
        
        for folder in folder_names:
            logging.info(f"Current video: {folder}")
            test_dataset = UVGTestDataset(
                args.test_path, 
                [folder], 
                gop_size=args.test_gop_size,
                skip_frames=args.test_skip_frames, 
                test_size=args.test_numbers
            )
            h, w, _  = test_dataset.orig_img_size
            
            logging.info(f"Shape: {h}x{w}")
            
            # To adjust the bidirectional scheme, we increase the batch size by 1
            test_loader = DataLoader(
                test_dataset, batch_size=args.test_gop_size+1, 
                shuffle=False, num_workers=args.workers, drop_last=True
            )
            
            decoded = {}
    
            # Loading videos in batches of form I-B-B-B-B-B-B-B-I
            for idx, gop in enumerate(test_loader):
                gop = gop.unsqueeze(1).to(device)
                
                
                if idx == 0:
                    dec0, size0 = image_compress(gop[0], i_model)
                    decoded[0] = dec0

                    uint8_real0 = float_to_uint8(gop[0][0, :, :h, :w].cpu().numpy())
                    uint8_dec_out0 = float_to_uint8(dec0[0, :, :h, :w].cpu().detach().numpy())
                    
                    psnr0 = PSNR(
                        MSE(uint8_dec_out0.astype(np.float64), uint8_real0.astype(np.float64)), 
                        data_range=255
                    )
                    
                    infographic.update("I", 0, args.i_qual, folder, psnr0, size0.item(), 
                                        float(h * w))
                
                if (idx + 1)%args.i_interval == 0:
                
                    dec_last, size_last = image_compress(gop[-1], i_model)
                    frame_type = "I"
                    frame_num = 0
                else:
                    pass
                
                decoded[coding_order[1]] = dec_last
                
                uint8_real_last = float_to_uint8(gop[-1][0, :, :h, :w].cpu().numpy())
                uint8_dec_out_last = float_to_uint8(dec_last[0, :, :h, :w].cpu().detach().numpy())
                
                psnr_last = PSNR(
                    MSE(uint8_dec_out_last.astype(np.float64), uint8_real_last.astype(np.float64)), 
                    data_range=255
                )
                
                infographic.update(
                    frame_type, frame_num, args.i_qual, folder, 
                    psnr_last, size_last.item(), float(h * w)
                )
                
                for order in coding_order_eff:
                    dec_frame, _, dec_size = b_model(
                        x_before=decoded[decoding_info[order][0]], 
                        x_current=gop[order],
                        x_after=decoded[decoding_info[order][1]],
                        train=False
                    )
                    decoded[order] = dec_frame

                    uint8_real = float_to_uint8(gop[order][0, :, :h, :w].cpu().numpy())
                    uint8_dec_out = float_to_uint8(dec_frame[0, :, :h, :w].cpu().detach().numpy())

                    cur_psnr = PSNR(
                        MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), 
                        data_range=255
                    )
                    frame_num = (idx % args.i_interval) * args.test_gop_size + order
                    
                    infographic.update(
                        "B", frame_num, args.i_qual, folder, cur_psnr, dec_size, float(h * w))
                    
                decoded = {0: dec_last}
            
            logging.info("*********************************")
            infographic.print_per_video_level(folder)
            logging.info("*********************************")
            infographic.print_per_video_level_frame_type(folder)
            logging.info("*********************************")
 
    return infographic
    

# ### Main Function


def plot_with_dots(bpp, psnr, label, c):
    plt.plot(bpp, psnr, ".", color=c, markersize=15, label=label)
    plt.plot(bpp, psnr, "-", color=c)


def main(args):
    # Build I model
    i_model = mbt2018_mean(args.i_qual, "mse", pretrained=True).to(device).float()

    for param in i_model.parameters():
        param.requires_grad = False
        
    i_model = i_model.eval()

    # Build B model
    b_model = Model()
    b_model.load_state_dict(
        torch.load(
            args.b_pretrained, 
            map_location=lambda storage, loc: storage
        )["state_dict"]
    )
    b_model = b_model.to(device).float()
    
    for param in b_model.parameters():
        param.requires_grad = False
            
    b_model = b_model.eval()

    infographic = test(
        b_model=b_model,
        i_model=i_model, 
        device=device, 
        args=args
    )
                    
    # Log to logfile if wanted
    if args.log_results:
        logging.info("-------------------------------")
    
        infographic.print_per_frame_type_level()
        
        logging.info("---------------------------------")
        infographic.print_per_level()
        logging.info("---------------------------------")
        logging.info("*********************************")
        infographic.print_per_level_frame_num()
        logging.info("*********************************")
        
        bpps, psnrs = [], []
        for bpp, psnr in infographic.average_bpp_psnr_dict.items():
            bpps.append(bpp)
            psnrs.append(psnr)
        
        fig = plt.figure(figsize= (13,13))
  
        elf_vc_psnr = [35.6, 36.50, 37.10, 37.60, 38.40, 38.8, 39.40]
        elf_vc_bpp = [0.04, 0.06, 0.08, 0.10, 0.16, 0.2, 0.28]
        
        proposed_psnr = [
            35.80746748445539, 36.08968679030873, 36.387034883902665, 36.7247531270879, 
            37.04453450374351, 37.33458740737803, 37.59928209526169, 37.8187249518319, 
            38.059522641651796, 38.315368515447496, 38.55869424920168, 38.777194053887996, 
            38.94942917796628, 39.05401996138734, 39.130509643847304, 39.179325261653915
        ]
        proposed_bpp = [
            0.04842646709468422, 0.05293394876977796, 0.05853089596925436, 0.06530656060656882, 
            0.07420733720931912, 0.08407375547296572, 0.09459414696863049, 0.10497012071681389,
            0.11821998015491471, 0.13568459237405384, 0.1563638581362788, 0.1776865851437145, 
            0.19834474796251989, 0.21249289542795538, 0.22453242022489403, 0.23343755025738502
        ]
        
        tfp_sub4_mc_noctx_psnr = [35.43, 36.54, 37.52, 38.45, 39.40]
        tfp_sub4_mc_noctx_bpp = [0.0457, 0.0648, 0.0950, 0.1722, 0.3074]
        
        plot_with_dots(
            bpps,
            psnrs,
            args.model_name,
            c="blue"
        )
        plot_with_dots(
            proposed_bpp,
            proposed_psnr,
            "proposed with 6 levels",
            c="red"
        )
        plot_with_dots(
            tfp_sub4_mc_noctx_bpp,
            tfp_sub4_mc_noctx_psnr,
            "tfp_sub4_mc_noctx",
            c="orange"
        )
        plot_with_dots(
            elf_vc_bpp,
            elf_vc_psnr,
            "elf_vc",
            c="black"
        )
        
        plt.grid()
        plt.ylabel("PSNR (dB)")
        plt.xlabel("bpp (bit/sec)")
        plt.legend()
        
        plt.savefig(args.model_name + ".png")
        
        
        infographic.save_excel(f"{args.model_name}.xlsx")

    print("------ Bits per pixel ------")
    print(bpps)
    print("----------- PSNR -----------")
    print(psnrs)

# In[ ]:


if __name__ == '__main__':
    main(args)
