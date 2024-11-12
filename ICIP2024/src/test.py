
#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from omegaconf import DictConfig
import torch
import numpy as np
import os
import sys
import pandas as pd
import warnings
import hydra

from natsort import natsorted
import glob
warnings.filterwarnings('ignore')
import logging

log = logging.getLogger(__name__)

from torch.utils.data import DataLoader, RandomSampler

from .utils import float_to_uint8, MSE, PSNR, calculate_distortion_loss
from .utils import prepare_frame, image_compress, load_model
from .utils import get_scales, get_order_typ_list, update_buffer, select_references
from .opt_helpers import get_best_down_ratio_prediction, get_best_down_ratio_compress


from .bd_rate import *

from src.model import elic
from src.model import m


def val_sequence_level(video_sequence, im_model, model, betas, device, order_list, typ_list, level): 
        
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
            


def validate_all(im_model, model, betas, device, levels, dataset):
    # Initialize an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Level', 'Sequence', 'Frame', 'Loss', 'PSNR', 'Size','Type'])

    for level in range(levels):
        for sequence, seq_info in dataset['sequences'].items():
            sequence_path = os.path.join(dataset['base_path'], sequence)


            frame_files = natsorted(glob.glob(f"{sequence_path}/*.{dataset['src_type']}"))
            frame_files = frame_files[:seq_info['num_frames']]


            order_list, typ_list = get_order_typ_list(dataset['gop'], len(frame_files))


            loss_list, psnr_list, size_list = val_sequence_level(
                frame_files, im_model, model, betas, device, order_list, typ_list, level
            )

          
            log.info(f"Level: {level} | Sequence: {sequence:20} | PSNR: {np.mean(psnr_list):2.2f} | Size: {np.mean(size_list):8.6f}")


            for frame_idx, (loss, psnr, size,type) in enumerate(zip(loss_list, psnr_list, size_list, typ_list)):
                results_df = results_df.append({
                    'Level': level,
                    'Sequence': sequence,
                    'Frame': frame_idx,
                    'Loss': loss,
                    'PSNR': psnr,
                    'Size': size,
                    'Type': type
                }, ignore_index=True)

    return results_df



def load_models(config:DictConfig):
    intra_models = []
    for i in range(config.levels):
        im = elic.ELIC()
        checkpoint = torch.load(config.intra_models[i])
        im.load_state_dict(checkpoint['state_dict'])
        im = im.to(config.device).float()
        im.eval()
        intra_models.append(im)
        
    model = m.FlowGuidedB().to(config.device).float()
    if config.pretrained:
        model.load_state_dict(torch.load(config.pretrained, map_location=lambda storage, loc: storage))
    model = model.eval()
    return intra_models, model


def run(cfg:DictConfig):
    intra_models, model = load_models(cfg)
    betas_mse = torch.tensor(cfg.betas_mse) * (255**2)
    betas_mse = betas_mse.to(cfg.device)
    device = cfg.device
    levels = cfg.levels
    dataset = cfg.dataset
    
    
    results_df = validate_all(intra_models, model, betas_mse, device, levels, dataset)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_df.to_csv(os.path.join(output_dir, cfg.results_path), index=False)


