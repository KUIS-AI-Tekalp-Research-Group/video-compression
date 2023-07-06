
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import glob
import os
import itertools
import warnings
warnings.filterwarnings('ignore')
import imageio
from natsort import natsorted
from compressai.zoo import mbt2018, mbt2018_mean, cheng2020_anchor
import math
import logging
import time
import random
import torch.nn.functional as F
import torch.optim as optim

from model import m

device = torch.device("cuda")

logging.basicConfig(filename="train.log",level=logging.INFO)


def normalize(tensor):
    norm = (tensor)/255.
    return norm

def float_to_uint8(image):
    clip = np.clip(image,0,1)*255.
    im_uint8 = np.round(clip).astype(np.uint8).transpose(1,2,0)
    return im_uint8

def MSE(gt,pred):
    mse = np.mean((gt-pred)**2)
    return mse

def PSNR(mse,data_range):
    psnr = 10*np.log10((data_range**2)/mse)
    return psnr

def calculate_distortion_loss(out,real):
    distortion_loss = torch.mean((out-real)**2)
    return distortion_loss



def train_video_loader(path="/datasets/vimeo_septuplet/sequences/"):
    videos = []
    folders = natsorted(glob.glob(path+"*"))    
    for f in folders:
        v = natsorted(glob.glob(f+"/*")) 
        for vid in v:
            videos.append(vid)
    return videos


def train_image_loader(videos):

    gop_video_batch = []
    
    for video in videos:
        gop_im_list = natsorted(glob.glob(video+"/*.png"))
        gop_video_batch.append(gop_im_list)
        
    return gop_video_batch


def prepare_train_data(gop_video_batch):
    
    X_train = []
    
    patch_size = 256
    size = 5
    length = 7

    for gop_ims in gop_video_batch:
    
        #size = random.choice([3,5,7])
        #length = len(gop_ims)        
        s = random.randint(0, length - size)
        #size = 7
        #gop_split = gop_ims
        
        gop_split = gop_ims[s:s+size]
        
        sample_im = imageio.imread(gop_split[0])
        
        x = random.randint(0, sample_im.shape[1] - patch_size)
        y = random.randint(0, sample_im.shape[0] - patch_size)
        
        img1 = imageio.imread(gop_split[0])
        img1 = img1[y:y+patch_size,x:x+patch_size].transpose(2,0,1)
        img2 = imageio.imread(gop_split[size//2])
        img2 = img2[y:y+patch_size,x:x+patch_size].transpose(2,0,1)
        img3 = imageio.imread(gop_split[-1])
        img3 = img3[y:y+patch_size,x:x+patch_size].transpose(2,0,1)

        img_concat = np.concatenate((img1,img2),axis=0)
        img_concat = np.concatenate((img_concat,img3),axis=0)

        X_train.append(img_concat)
    
    X_train = np.array(X_train)
        
    return X_train


def test_image_loader(path="/datasets/UVG/full_test/"):
    v = ["beauty", "bosphorus", "honeybee", "jockey", "ready", "shake", "yatch"]
    videos = [path+i for i in v]
    images_list = []
    
    for video in videos:
        l = natsorted(glob.glob(video+"/*.png"))[:9]
        images_list.append(l)
    return images_list

def image_compress(im, compressor):
    out1 = compressor(im)
    dec1 = out1["x_hat"]
    size_image = sum(
        (torch.log(likelihoods).sum() / (-math.log(2)))
        for likelihoods in out1["likelihoods"].values()
        )
    
    return dec1, size_image

def b_compress(im_before, im_current, im_after, model, train=True):
    if train:
        dec_current, dec_rate = model.forward(im_before, im_current, im_after, True)
        return dec_current, dec_rate 
    else:
        dec_current, dec_rate, dec_size = model.forward(im_before, im_current, im_after, False)
        return dec_current, dec_rate, dec_size


def pad(im):
    (m,c,w,h) = im.size()

    p1 = (64 - (w % 64)) % 64
    p2 = (64 - (h % 64)) % 64
    
    pad = nn.ReflectionPad2d(padding=(0, p2, 0, p1))
    return pad(im)

def process_frame(frame_path):
    x = imageio.imread(frame_path).transpose(2,0,1)
    (c,h,w) = x.shape
    x = x.reshape(1,c,h,w)
    x = normalize(torch.from_numpy(x).float())
    x  = pad(x)
    return x


def train_one_step(im_batch,model,optimizer,aux_optimizer,image_compressor):
    alpha = 3141.
    beta = 1.
    model = model.train()
    
    X_train = prepare_train_data(im_batch)
    X_train = normalize(torch.from_numpy(X_train).to(device).float())
    
    dec_x1, _ = image_compress(X_train[:, 0:3], image_compressor)
    x2 = X_train[:, 3:6]
    dec_x3, _ = image_compress(X_train[:, 6:9], image_compressor)
    
    dec_x2, rate_x2 = b_compress(dec_x1, x2, dec_x3, model, True)
    
    dl2 = calculate_distortion_loss(dec_x2, x2)
    rl2 = rate_x2
    
    dist_loss = alpha*dl2
    rate_loss = beta*rl2
    
    loss = dist_loss + rate_loss
    
    optimizer.zero_grad()
    aux_optimizer.zero_grad()
    
    aux_loss = (model.mv_compressor.aux_loss() + model.residual_compressor.aux_loss())/2.
    
    loss.backward()
    aux_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    aux_optimizer.step()
    
    return dist_loss.item(),rate_loss.item(),loss.item(),aux_loss.item()



            
    
def save_model(model,child,name):
    state = {
        "state_dict" : getattr(model, child).state_dict()
    }
    torch.save(state,name+".pth")
    
def save_all_model(model):
    state = {
        "state_dict" : model.state_dict()
    }
    torch.save(state,"compression.pth")

def save_optimizer(optimizer, name):
    state = {
        "state_dict" : optimizer.state_dict()
    }
    torch.save(state, name+".pth")


def main():   
    #np.random.seed(111)
    #torch.manual_seed(222)
    #random.seed(333)
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    image_compressor = mbt2018_mean(quality=8, metric="mse", pretrained=True).to(device).float()
    image_compressor = image_compressor.eval()
    
    total_train_step = 1000000
    train_step = 5000
    
    learning_rate = 1.e-4
    model = m.Model()
    #model.load_state_dict(torch.load("old.pth", map_location=lambda storage, loc: storage)["state_dict"])
    model = model.to(device).float()

    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=learning_rate)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("number of compress parameters: "+str(params))
        

    videos = train_video_loader()

    batch_size = 4
    
    step_aux_loss = 0
    step_train_loss = 0
    step_distortion_loss = 0
    step_rate_loss = 0
    
    best_test_loss = 10**10
    
    time_start = time.time()
    
       
    for minibatch_processed in range(1,total_train_step+1):
        
        gop_video_batch = random.sample(videos, batch_size)
        gop_im_batch = train_image_loader(gop_video_batch)
        
        total_distortion_loss,total_rate_loss,loss,aux_loss = train_one_step(gop_im_batch,model,optimizer,aux_optimizer,image_compressor)
        
        step_aux_loss += aux_loss
        step_train_loss += loss
        step_distortion_loss += total_distortion_loss
        step_rate_loss += total_rate_loss 
        
                
        if minibatch_processed % train_step == 0:
            
            save_all_model(model)
            #save_optimizer(optimizer, "optimizer")
            #save_optimizer(aux_optimizer, "aux_optimizer")


            logging.info("learning rate: "+str(get_lr(optimizer)))
            logging.info("iterations: "+str(minibatch_processed))
            logging.info("distortion loss: "+str(step_distortion_loss/train_step))
            logging.info("rate loss: "+str(step_rate_loss/train_step))
            logging.info("train loss: "+str(step_train_loss/train_step))
            logging.info("aux loss: "+str(step_aux_loss/train_step))
            logging.info("**************")
            
            
            step_aux_loss = 0
            step_train_loss = 0
            step_distortion_loss = 0
            step_rate_loss = 0
        
        
if __name__=='__main__':
    main()
