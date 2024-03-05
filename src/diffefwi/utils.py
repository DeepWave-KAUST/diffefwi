"""
Utility functions.
@hatsyim
"""

import math
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import butter, lfilter

def set_seed(seed):
    """
    :param seed: An integer of random number.

    :return: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def butter_filter(data, lowcut, highcut, fs, order=6, filt_type='band'):
    """
    Define a butter filter.

    :params data: A time series array.
    :params lowcut: Low frequency band (Hz).
    :params highcut: High frequency band (Hz).
    :params fs: Sampling frequency (Hz).
    
    :return: Filtered data.
    """
    
    b, a = butter(order, [lowcut, highcut], fs=fs, btype=filt_type)
    y = lfilter(b, a, data)
    return y

def snr(x, x_est):
    """
    Compute the signal-to-noise ratio (SNR) in dB.
    
    :params x: Original signal
    :params x_est: Estimated signal
    
    :return: SNR in dB
    """
    
    return 10.0 * np.log10(np.linalg.norm(x) / np.linalg.norm(x - x_est))

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    
    :return: an [N x dim] Tensor of positional embeddings.
    """ 
    
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half,
                                             dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], 
                              dim=-1)
    return embedding

def norm_layer(channels):
    """
    Create group normalization layer.
    
    :param channels: int number of channels.
    
    :return: PyTorch module.
    """
    
    return nn.GroupNorm(32, channels)

def linear_beta_schedule(timesteps):
    """
    Linear noise scheduler.
    
    :param timesteps: array of time steps.
    
    :return: PyTorch array.
    """
    
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    #beta_end = scale * 0.0005
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise scheduler in https://arxiv.org/abs/2102.09672.
    
    :param timesteps: array of time steps.
    
    :return: PyTorch array.
    """
    
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def vp_to_vs(vp):
    """
    Convert compressional to shear velocity.
    
    :param vp: a compressional velocity array.
    
    :return vs: a shear velocity array.
    """
    
    return vp/np.sqrt(3)

def vp_to_rho(vp):
    """
    Convert compressional to shear velocity.
    
    :param vp: a compressional velocity array.
    
    :return rho: a density array.
    """
    
    return 0.31*(vp)**(0.25)

def normalize_vp(vp, vmax=5000, vmin=3000):
    """
    Normalize compressional velocity.
    
    :param vp: a compressional velocity array.
    :param vmax: maximum vp.
    :param vmin: minimum vp.
    
    :return vp: a normalized vp [0-1].
    """
    
    return (vp - vmin)/(vmax-vmin)*2.0 - 1.0

def denormalize_vp(vp, vmax=5000, vmin=3000):
    """
    Denormalize compressional velocity.
    
    :param vp: a normalized compressional velocity array [0-1].
    :param vmax: maximum vp.
    :param vmin: minimum vp.
    
    :return vp: a denormalized vp.
    """
    
    return (vp + 1.0)/2.0*(vmax-vmin)+vmin

def normalize_vs(vs, vmax=2887, vmin=1732):
    """
    Denormalize shear velocity.
    
    :param vs: a shear velocity array.
    :param vmax: maximum vs.
    :param vmin: minimum vs.
    
    :return vs: a normalized vs [0-1].
    """
    
    return (vs - vmin)/(vmax-vmin)*2.0 - 1.0

def denormalize_vs(vs, vmax=2887, vmin=1732):
    """
    Denormalize shear velocity.
    
    :param vs: a normalized shear velocity array [0-1].
    :param vmax: maximum vs.
    :param vmin: minimum vs.
    
    :return vs: a denormalized vs.
    """
    
    return (vs + 1.0)/2.0*(vmax-vmin)+vmin

def normalize_rho(rho, vmax=2607, vmin=2294):
    """
    Normalize density.
    
    :param vs: a density array.
    :param vmax: maximum rho.
    :param vmin: minimum rho.
    
    :return vs: a normalized rho [0-1].
    """
    
    return (rho - vmin)/(vmax-vmin)*2.0 - 1.0

def denormalize_rho(rho, vmax=2607, vmin=2294):
    """
    Denormalize density.
    
    :param vs: a normalized density array [0-1].
    :param vmax: maximum rho.
    :param vmin: minimum rho.
    
    :return vs: a denormalized rho.
    """
    
    return (rho + 1.0)/2.0*(vmax-vmin)+vmin

def split_data_to_size(data, size, stride):
    """
    Extract patches of images.

    :param data: Input images.
    :param size: Kernel size.
    :param stride: Stride size.

    :return: Patch of images.
    """
    b, c, h, w = data.shape
    h_split, w_split =  (h-size[0]+1)//stride[0], (w-size[1]+1)//stride[1]
    w_split_add = (w-size[1]+1)%stride[1]
    b_new, c_new, h_new, w_new = h_split * (w_split + w_split_add), c, size[0], size[1]
    data_split = torch.zeros(b_new,c_new,h_new,w_new).to(data.device)
    for i in range(h_split):
        for j in range(w_split):
            data_split[i*w_split+j,:,:,:] = data[:,:,i*stride[0]:i*stride[0]+h_new,j*stride[1]:j*stride[1]+w_new]
    for i in range(w_split_add):
        data_split[(h_split-1)*w_split+w_split+i,:,:,:] = data[:,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h_new,w_split*stride[1]+i:w_split*stride[1]+i+w_new]
    return data_split.reshape(-1,c_new,h_new,w_new)
    
def merge_data_to_size(data, size, stride):
    """
    Extract patches of images.

    :param data: Input images.
    :param size: Kernel size.
    :param stride: Stride size.

    :return: Patch of images.
    """
    b, c, h, w = data.shape
    h_split, w_split =  (size[0]-h+1)//stride[0], (size[1]-w+1)//stride[1]
    w_split_add = (size[1]-w+1)%stride[1]
    b_new, c_new, h_new, w_new = 1, c, size[0], size[1]
    data_merge = torch.zeros(b_new,c_new,h_new,w_new).to(data.device)
    mask = torch.zeros((b_new*c, h_new, w_new)).reshape(b_new,c,h_new,w_new).to(data.device)
    for i in range(h_split):
        for j in range(w_split):
            data_merge[0,:,i*stride[0]:i*stride[0]+h,j*stride[1]:j*stride[1]+w] += data[i*w_split+j,:,:,:]
            mask[0,:,i*stride[0]:i*stride[0]+h,j*stride[1]:j*stride[1]+w] +=1
    for i in range(w_split_add):
        data_merge[0,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h,w_split*stride[1]+i:w_split*stride[1]+i+w] += data[w_split+i,:,:,:]
        mask[0,:,(h_split-1)*stride[0]:(h_split-1)*stride[0]+h,w_split*stride[1]+i:w_split*stride[1]+i+w] += 1
    return data_merge.reshape(-1,c,h_new,w_new)/mask