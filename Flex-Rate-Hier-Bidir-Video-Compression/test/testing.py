
# # 1. Test code

import torch
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import imageio

from compressai.zoo import mbt2018_mean

    
import argparse
import logging
import time
import random
import math
import matplotlib.pyplot as plt
from torch import optim

from torch.utils.data import DataLoader, RandomSampler

model_path = os.path.abspath('..')
sys.path.insert(1, model_path)

from b_model.b_model import BidirFlowRef

from utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from utils import UVGTestDataset, load_model, compressai_image_compress
from utils import TestInfographic

torch.backends.cudnn.benchmark = True


# Argument parser
parser = argparse.ArgumentParser()

# Hyperparameters, paths and settings are given
# prior the test
parser.add_argument("--project_name", type=str, default="ICIP2022_test")                        # Project name
parser.add_argument("--model_name", type=str, default="flex_rate_gained")                       # Model name
          
parser.add_argument("--test_path", type=str, default="/path/to/uvg/dataset/")                   # Dataset paths

parser.add_argument("--test_gop_size", type=int, default=16)                                    # Test gop sizes
parser.add_argument("--i_interval", type=int, default=16)                                       # Test i compression interval
parser.add_argument("--test_skip_frames", type=int, default=1)                                  # Test skip frames
parser.add_argument("--test_numbers", type=int, default=None)                                   # How many times to test on a video

parser.add_argument("--device", type=str, default="cuda")                                       # device "cuda" or "cpu"
parser.add_argument("--workers", type=int, default=4)                                           # number of workers

parser.add_argument("--levels", type=int, default=4)                                            # Number of points on rate-distortion curve

parser.add_argument("--b_pretrained", type=str, default="../pretrained.pth")                    # Load model from this file

parser.add_argument("--log_results", type=bool, default=True)                                   # Store results in log

args = parser.parse_args()

args.save_name = args.model_name

logging.basicConfig(filename= args.save_name + "_test.log", level=logging.INFO)

args.i_interval /= args.test_gop_size

# Frame order for decoding
coding_order = [0, 16, 8, 4, 2, 1, 3, 6, 5, 7, 12, 10, 9, 11, 14, 13, 15]
# prev_frame, future_frame, frame_level
decoding_info = {8: [0, 16], 4: [0, 8], 2: [0, 4], 1: [0, 2], 3: [2, 4], 6: [4, 8], 5: [4, 6], 7: [6, 8],
                 12: [8, 16], 10: [8, 12], 9: [8, 10], 11: [10, 12], 14: [12, 16], 13: [12, 14], 15: [14, 16]}
                 
hier_levels = {8: 0, 4: 1, 2: 2, 1: 3, 3: 3, 6: 2, 5: 3, 7: 3,
               12: 1, 10: 2, 9: 3, 11: 3, 14: 2, 13: 3, 15: 3}
               
args.num_i = (5, 6, 7, 8)
args.levels_intervals = [(0, 1.)]            # dummy variable, used just for the excel output
""" qualities = [(5, {0: (1, 1.), 1: (0, 0.33), 2: (0, 0.66), 3: (0, 1.)}), (6, {0: (1, 0.66), 1: (1, 1.), 2: (0, 0.33), 3: (0, 0.66)}), 
             (6, {0: (1, 0.33), 1: (1, 0.66), 2: (1, 1.), 3: (0, 0.33)}), (6, {0: (2, 1.), 1: (1, 0.33), 2: (1, 0.66), 3: (1, 1.)}), 
             (7, {0: (2, 0.66), 1: (2, 1.), 2: (1, 0.33), 3: (1, 0.66)}), (7, {0: (2, 0.33), 1: (2, 0.66), 2: (2, 1.), 3: (1, 0.33)}),
             (7, {0: (3, 1.), 1: (2, 0.33), 2: (2, 0.66), 3: (2, 1.)}), (8, {0: (3, 1.), 1: (3, 1.), 2: (2, 0.33), 3: (2, 0.66)}), 
             (8, {0: (3, 1.), 1: (3, 1.), 2: (3, 1.), 3: (2, 0.33)}), (8, {0: (3, 1.), 1: (3, 1.), 2: (3, 1.), 3: (3, 1.)})] """
qualities = [(5, {0: (1, 1.), 1: (0, 0.33), 2: (0, 0.66), 3: (0, 1.)}), (6, {0: (1, 0.66), 1: (1, 1.), 2: (0, 0.33), 3: (0, 0.66)}), 
             (6, {0: (1, 0.33), 1: (1, 0.66), 2: (1, 1.), 3: (0, 0.33)}), (6, {0: (2, 1.), 1: (1, 0.33), 2: (1, 0.66), 3: (1, 1.)}), 
             (7, {0: (2, 0.66), 1: (2, 1.), 2: (1, 0.33), 3: (1, 0.66)}), (7, {0: (2, 0.33), 1: (2, 0.66), 2: (2, 1.), 3: (1, 0.33)}),
             (7, {0: (3, 1.), 1: (2, 0.33), 2: (2, 0.66), 3: (2, 1.)}), (8, {0: (3, 1.), 1: (3, 1.), 2: (3, 1.), 3: (2, 0.33)})]


# ### Test Function

def mbt_image_compress(im_batch, model, n, l):
    _, _, h, w = im_batch.shape
    num_pixels = h * w
    
    output = model(
        im_batch, 
        n=n,
        l=l,
    )
    
    size = sum(
        (torch.log(likelihoods).sum(dim=(1, 2, 3)) / (-math.log(2)))
        for likelihoods in output["likelihoods"].values()
    )
    rate = size / num_pixels
    
    dec = output["x_hat"]
    return dec, rate, size


def test(b_model, i_models, device, args):
    """
    test_loader: Test loader for UVG
    model: Composite B-frame compressor model
    device: cuda or cpu
    """
    
    with torch.no_grad():
        coding_order_eff = coding_order[2:]
        
        folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]

        infographic = TestInfographic(args.levels_intervals, folder_names)        
        
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
            test_loader = DataLoader(test_dataset, batch_size=args.test_gop_size+1, shuffle=False, num_workers=args.workers, drop_last=True)
            
            decoded = {}
    
            # Loading videos in batches of form I-B-B-B-...-B-B-B-B-I
            for idx, gop in enumerate(test_loader):
                gop = gop.unsqueeze(1).to(device)
                
                for qual in qualities:
                    i_qual, b_qual_dict = qual
                    i_model = i_models[i_qual]
                     
                    level, interval = b_qual_dict[3]
                 
                    if idx == 0:
                        decoded[(level, interval)] = {}
                        dec0, size0 = compressai_image_compress(gop[0], i_model)
                        decoded[(level, interval)][0] = dec0
                       
                        uint8_real0 = float_to_uint8(gop[0][0, :, :h, :w].cpu().numpy())
                        uint8_dec_out0 = float_to_uint8(dec0[0, :, :h, :w].cpu().detach().numpy())
                        
                        psnr0 = PSNR(
                            MSE(uint8_dec_out0.astype(np.float64), uint8_real0.astype(np.float64)), 
                            data_range=255
                        )
                        
                        infographic.update("I", 0, level, interval, folder, psnr0, size0.item(), 
                                           uint8_real0.shape[0] * uint8_real0.shape[1])
                    
                    if (idx + 1)%args.i_interval == 0:
                    
                        dec_last, size_last = compressai_image_compress(gop[-1], i_model)
                        frame_type = "I"
                        frame_num = 0
                    else:
                        pass
                    
                    decoded[(level, interval)][coding_order[1]] = dec_last
                    
                    uint8_real_last = float_to_uint8(gop[-1][0, :, :h, :w].cpu().numpy())
                    uint8_dec_out_last = float_to_uint8(dec_last[0, :, :h, :w].cpu().detach().numpy())
                    
                    psnr_last = PSNR(
                        MSE(uint8_dec_out_last.astype(np.float64), uint8_real_last.astype(np.float64)), 
                        data_range=255
                    )
                    
                    infographic.update(frame_type, frame_num, level, interval, folder, psnr_last, size_last.item(), 
                                       uint8_real_last.shape[0] * uint8_real_last.shape[1])
    
                    for order in coding_order_eff:
                        output = b_model(
                            x_before=decoded[(level, interval)][decoding_info[order][0]], 
                            x_current=gop[order],
                            x_after=decoded[(level, interval)][decoding_info[order][1]],
                            n=[b_qual_dict[hier_levels[order]][0]],
                            l=b_qual_dict[hier_levels[order]][1],
                            train=False
                        )
                        decoded[(level, interval)][order] = output["x_hat"]
    
                        uint8_real = float_to_uint8(gop[order][0, :, :h, :w].cpu().numpy())
                        uint8_dec_out = float_to_uint8(output["x_hat"][0, :, :h, :w].cpu().detach().numpy())
    
                        cur_psnr = PSNR(
                            MSE(uint8_dec_out.astype(np.float64), uint8_real.astype(np.float64)), 
                            data_range=255
                        )
                        frame_num = (idx % args.i_interval) * args.test_gop_size + order
    
                        infographic.update("B", frame_num, level, interval, folder, cur_psnr, 
                                           output["size"].squeeze(0).item(), 
                                           uint8_real.shape[0] * uint8_real.shape[1])
                        
                    decoded[(level, interval)] = {0: dec_last}
            
            logging.info("*********************************")
            infographic.print_per_video_level(folder)
            logging.info("*********************************")
            infographic.print_per_video_level_frame_type(folder)
            logging.info("*********************************")
 
    return infographic
    

def plot_with_dots(bpp, psnr, label, c):
    plt.plot(bpp, psnr, ".", color=c, markersize=15, label=label)
    plt.plot(bpp, psnr, "-", color=c)


def main(args):

    device = torch.device(args.device)
    
    # Build I model
    i_models = {q: mbt2018_mean(q, "mse", pretrained=True).to(device).float() 
                for q in args.num_i}
    
    for q in i_models.keys():
        for param in i_models[q].parameters():
            param.requires_grad = False
            
        i_models[q] = i_models[q].eval()
    
    # Build B model
    b_model = BidirFlowRef(n=args.levels, N=128).to(device).float()
    
    if args.b_pretrained:
        checkpoint = torch.load(args.b_pretrained, map_location=device)
        b_model = load_model(b_model, checkpoint, exceptions=[])
     
    b_model = b_model.eval()
    
    time_start = time.perf_counter()

    infographic = test(
        b_model=b_model,
        i_models=i_models, 
        device=device, 
        args=args
    )

    time_end = time.perf_counter()
    duration = time_end - time_start

                    
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
  
        elf_vc_psnr = [35.6, 36.50, 37.10, 37.60, 38.40, 38.78, 39.40]
        elf_vc_bpp = [0.04, 0.06, 0.08, 0.10, 0.16, 0.2, 0.28]
        
        tfp_nosub_mc_noctx_psnr = [35.34, 36.52, 37.50, 38.42, 39.38]
        tfp_nosub_mc_noctx_bpp = [0.0461, 0.0691, 0.1071, 0.1870, 0.3321]
        
        tfp_sub4_mc_noctx_psnr = [35.43, 36.54, 37.52, 38.45, 39.40]
        tfp_sub4_mc_noctx_bpp = [0.0457, 0.0648, 0.0950, 0.1722, 0.3074]
        
        plot_with_dots(
            bpps,
            psnrs,
            args.model_name,
            c="blue"
        )
        plot_with_dots(
            tfp_nosub_mc_noctx_bpp,
            tfp_nosub_mc_noctx_psnr,
            "tfp_nosub_mc_noctx",
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


if __name__ == '__main__':
    main(args)

