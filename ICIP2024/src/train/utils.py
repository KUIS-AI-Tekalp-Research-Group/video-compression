#!/usr/bin/env python
# coding: utf-8

# # 2. Utils

# In[ ]:


import os
import torch
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
from natsort import natsorted
import glob
import random
import sys
import math
import torch.nn as nn
import logging



    
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
    #order = [8, 4, 2, 6, 1, 3, 5, 7]
    order = [16, 8, 4, 12, 2, 14, 6, 10, 1, 15, 3, 13, 5, 11, 7, 9]
    #order = [4, 2, 1, 3]
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

    pad = nn.ZeroPad2d(padding=(0, p2, 0, p1))
    return pad(im).squeeze(0)
        

# ### Training & Test Video & Image Datasets


from torch.utils.data import Dataset


class VimeoTrainDataset(Dataset):
    """Dataset for custom vimeo"""

    def __init__(self, data_path, patch_size, rng, dtype=".png"):
        """
        data_path: path to folders of videos,
        patch_size: size to crop for training,
        gop_size: GoP size,
        skip_frames: do we skip frames (int),
        num_frames: whether we limit the number of frames in the GoP,
        rng: random number generator,
        dype: png or jpeg
        """
    
        self.cropper = torchvision.transforms.RandomCrop(size=patch_size)
        self.options = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1],
                        [2, 3, 4, 5, 6], [6, 5, 4, 3, 2]]
        
        """self.options = [[0, 2, 4], [4, 2, 0], [1, 3, 5], [5, 3, 1],
                        [2, 4, 6], [6, 4, 2], [0, 3, 6], [6, 3, 0]]"""
        self.fb = 1

        self.data_path = data_path

        # Pick the videos with sufficient resolution
        videos = []
        folders = natsorted(glob.glob(data_path + "*"))
        for folder in folders:
            videos += natsorted(glob.glob(folder + "/*"))
        
        self.videos = videos
        
        self.patch_size = patch_size
        self.dtype = dtype
        
        # Random number generator for reproducability
        self.rng = rng
        
        # How many frames to take

        self.dataset_size = len(self.videos)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        video = self.videos[item]
        video_im_list = natsorted(glob.glob(video + "/*." + self.dtype))
        
        selected_frames = self.rng.sample(self.options, self.fb)[0]
        i1, i2, i3, i4, i5 = selected_frames[0], selected_frames[1], selected_frames[2], selected_frames[3], selected_frames[4]   

        im1 = torchvision.io.read_image(video_im_list[i1])
        im2 = torchvision.io.read_image(video_im_list[i2])
        im3 = torchvision.io.read_image(video_im_list[i3])
        im4 = torchvision.io.read_image(video_im_list[i4])
        im5 = torchvision.io.read_image(video_im_list[i5])

        video_split = torch.cat([im1, im2, im3, im4, im5], dim=0)
        
        video_split = self.cropper(video_split)        
        video_split = normalize(video_split)
                
        return video_split

class UVGTestDataset(Dataset):
    """Dataset for UVG"""

    def __init__(self, data_path, video_names, gop_size, skip_frames, test_size=2):
        """
        data_path: path to folders of videos,
        video_name: video name (e.g. beauty),
        skip_frames: do we skip frames (int, e.g. 1),
        """
        # Get the frame paths for each frame
        self.data_path = data_path
        self.skip_frames = skip_frames
        self.gop_size = gop_size
        self.test_size = test_size
        self.frames = []
        
        for video_name in video_names:
            video = data_path + video_name
            frames = natsorted(glob.glob(video + "/*.png"))[:test_size*gop_size+1]
            
            for idx, frame in enumerate(frames):
                self.frames.append(frame)
                if (idx % gop_size == 0) and (idx != 0) and (idx // gop_size != test_size):
                    self.frames.append(frame)
        
        self.dataset_size = len(self.frames)
        self.orig_img_size = imageio.imread(self.frames[0]).shape
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):        
        frame = self.frames[item]
        
        im = imageio.imread(frame).transpose(2, 0, 1)
        im = normalize(torch.from_numpy(im)).unsqueeze(0)
        im = pad(im)
        
        return im
    

def image_compress(im, compressors, n):
    out = compressors[n](im)
    dec = out["x_hat"]
    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out["likelihoods"].values()
    )

    return dec, size_image




# ### Save and load pmodel

# In[12]:


def save_model(model, optimizer, aux_optimizer, scheduler, num_iter, exceptions, save_name="checkpoint.pth"):
    """
    Save a model with its optimizer, aux_optimizer, scheduler and # of iteration info.
    If some of them are not desired, give None as input instead of it
    """
    
    save_dict = {}
    if optimizer:
        save_dict["optimizer"] = optimizer.state_dict()
    if aux_optimizer:
        save_dict["aux_optimizer"] = aux_optimizer.state_dict()
    
    if scheduler:
        save_dict["scheduler"] = scheduler.state_dict()
        
    if num_iter:
        save_dict["iter"] = num_iter
    
    for child, module in model.named_children():
        # If we don't want to save a child, we skip it
        if child in exceptions:
            continue
        save_dict[child] = module.state_dict()
        logging.info("Saved " + child + " at " + save_name)
        
    torch.save(save_dict, save_name)




def configure_seeds(random_seed=None, torch_seed=None):
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    if torch_seed is None:
        torch_seed = torch.seed()
    else:
        torch.manual_seed(torch_seed)
    
    rng = random.Random(random_seed)
    logging.info("Random library seed: " + str(random_seed))
    logging.info("PyTorch library seed: " + str(torch_seed))
    
    return rng

# In[14]:


def configure_optimizers(model, args):
    (lr, aux_lr, patience, min_lr) = args
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    # Use list of tuples instead of dict to be able to later check the elements are unique and there is no intersection
    parameters = []
    aux_parameters = []
    parameter_dict = {}
    for name, param in model.named_parameters():
        parameter_dict[name] = param
        if not name.endswith(".quantiles"):
            parameters.append((name, param))
        else:
            aux_parameters.append((name, param))
    
    aux_param_set = set(p for n, p in aux_parameters)
    num_aux_params = sum([np.prod(p.size()) for p in aux_param_set])
    
    logging.info("There are " + str(num_aux_params) + " aux_parameters")
    
    # Make sure we don't have an intersection of parameters
    """parameters_name_set = set(n for n,p in parameters)
    aux_parameters_name_set = set(n for n, p in aux_parameters)
    assert len(parameters) == len(parameters_name_set)
    assert len(aux_parameters) == len(aux_parameters_name_set)
    
    inter_params = parameters_name_set & aux_parameters_name_set
    union_params = parameters_name_set | aux_parameters_name_set
    assert len(inter_params) == 0
    assert len(union_params) - len(parameter_dict.keys()) == 0"""

    optimizer = optim.Adam((p for (n, p) in parameters if p.requires_grad),
                               lr=lr)

    
    aux_optimizer = optim.Adam((p for (n, p) in aux_parameters if p.requires_grad),
                               lr=aux_lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=patience, min_lr=min_lr)

    return optimizer, aux_optimizer, scheduler


