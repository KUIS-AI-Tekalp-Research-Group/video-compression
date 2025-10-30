#!/usr/bin/env python
# coding: utf-8

import torch

log_name = "logger"

random_seed = None
torch_seed = None

intra_model_path = "intra_weights"

train_path = "vimeo_septuplet/sequences/"
val_path = "UVG/frames/"

intra_size = 16


change_lr_step = 500000
total_train_step = 750000
stage1_train_step = 350000
train_step = 10000
total_val_frames = 17

learning_rate = 1.e-4
aux_learning_rate = 1.e-3
min_lr = 5.e-6
patience = 10
batch_size = 8
patch_size = 256

workers = 4
device = torch.device("cuda")

model_save_dir = "../model.pth"
optimizer_save_dir = "../optimizer.pth"
aux_optimizer_save_dir = "../aux_optimizer.pth"


folder_names = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yacht"]

betas_mse = torch.tensor([0.0056*(255**2), 0.0107*(255**2), 0.0207*(255**2), 
                               0.0400*(255**2), 0.0772*(255**2)]).to(device)  
levels = betas_mse.shape[0]                                                          

pretrained_model = None
pretrained_optimizer = None
pretrained_aux_optimizer = None

