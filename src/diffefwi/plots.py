import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

def create_images_grid(images, titles, cmap='jet', vmin=None, vmax=None, colorbar_label='', fig_size=(10, 10), file_name=None, **kwargs):
    """
    Create a square grid of images with colorbars.

    :param images: List of 2D NumPy arrays representing the images.
    :param titles: List of strings representing the titles of the images.
    :param colorbar_label: Label for the colorbar. Defaults to an empty string.
    :param fig_size: Figure size in inches. Defaults to (10, 10).
    """
    n = len(images)  # Number of images
    rows = int(np.sqrt(n))  # Number of rows in the grid
    cols = int(np.ceil(n / rows))  # Number of columns in the grid

    # Create the figure and axes
    fig, axs = plt.subplots(rows, cols, figsize=fig_size)

    # Flatten the axes if necessary
    if n == 1:
        axs = np.array([axs])

    # Iterate over the images and their titles
    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // cols  # Row index
        col = i % cols  # Column index

        # Display the image
        axs[row, col].imshow(image, cmap=cmap)
        axs[row, col].set_title(title)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

        # Add a colorbar to the right
        img_plot = axs[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', **kwargs)
        cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])  # Colorbar position
        
        if i==n-1:
            fig.colorbar(img_plot, cax=cax)
            cax.set_ylabel(colorbar_label)

    # Adjust the spacing and layout
    # fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    # Show the plot
    plt.show()

def plot_gather(obs_data, syn_data, fig_name=None, save_dir=None, **kwargs):
    """
    Visualize an interleave shot gathers.
    :param obs_data: Observed data array of size [num_timestep x num_receivers].
    :param obs_data: Generated data array of size [num_timestep x num_receivers].
    :param fig_name: Figure name.
    :param save_dir: Save folder directory.
    :return: matplotlib figure.
    """

    selected_shot = torch.cat((
    torch.flipud(syn_data) / torch.max(torch.abs(syn_data)), 
    obs_data / torch.max(torch.abs(obs_data)),
    torch.flipud(syn_data) / torch.max(torch.abs(syn_data)))
    )

    vmin, vmax = torch.quantile(selected_shot.detach().cpu(),
                                torch.tensor([0.05, 0.95]))
    
    _, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
    ax.imshow(
        selected_shot.detach().cpu().numpy().T, 
        aspect='auto',
        vmin=vmin, vmax=vmax, **kwargs
    )
    ax.set_ylabel("Time Sample")
    ax.set_xlabel("Shot")
    
    if save_dir is not None: 
        plt.savefig(save_dir+fig_name)

def plot_moduli(vp, vs, rho, fig_name=None, save_dir=None, distance_unit='km', **kwargs):
    """
    Visualize a 3 by 1 images.
    :param vp: Compressional velocity array of size [nx x nz].
    :param vs: Shear velocity array of size [nx x nz].
    :param rho: Density array of size [nx x nz].
    :param fig_name: Figure name.
    :param save_dir: Save folder directory.
    :return: matplotlib figure.
    """
    
    plt.figure(figsize=(15,2))
    
    ax0 = plt.subplot(1,3,1)
    plt.imshow(vp, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.ylabel('Depth ('+distance_unit+')', fontsize=14)
    plt.title('P-wave')
    cbar = plt.colorbar()
    cbar.set_label('Vp (m/s)')
    
    plt.subplot(1,3,2, sharey=ax0)
    plt.imshow(vs, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.title('S-wave')
    cbar = plt.colorbar()
    cbar.set_label('Vs (m/s)')
    
    plt.subplot(1,3,3, sharey=ax0)
    plt.imshow(rho, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.title('Density')
    cbar = plt.colorbar()
    cbar.set_label('Rho (g/cc)')
    
    if save_dir is not None: 
        plt.savefig(save_dir+fig_name)

def plot_ratios(vpvs, vprho, vsrho, fig_name=None, distance_unit='km', save_dir=None, **kwargs):
    """
    Visualize a 3 by 1 images.
    :param vp: Compressional velocity array of size [nx x nz].
    :param vs: Shear velocity array of size [nx x nz].
    :param rho: Density array of size [nx x nz].
    :param fig_name: Figure name.
    :param save_dir: Save folder directory.
    :return: matplotlib figure.
    """
    
    plt.figure(figsize=(15,2))
    
    ax0 = plt.subplot(1,3,1)
    plt.imshow(vp, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.ylabel('Depth ('+distance_unit+')', fontsize=14)
    plt.title('P-wave')
    cbar = plt.colorbar()
    cbar.set_label('Vp (m/s)')
    
    plt.subplot(1,3,2, sharey=ax0)
    plt.imshow(vs, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.title('S-wave')
    cbar = plt.colorbar()
    cbar.set_label('Vs (m/s)')
    
    plt.subplot(1,3,3, sharey=ax0)
    plt.imshow(rho, aspect='auto', **kwargs)
    plt.xlabel('Distance ('+distance_unit+')', fontsize=14)
    plt.title('Density')
    cbar = plt.colorbar()
    cbar.set_label('Rho (g/cc)')
    
    if save_dir is not None: 
        plt.savefig(save_dir+fig_name)

import torch

def extract_patches_2d(x, kernel_size, padding=0, stride=1, dilation=1):
    """
    Extracting patches from 2D images.
    :param kernel_size: Integer of window of patch size.
    :param padding: Integer of padding of patch size.
    :param stried: Integer of stried of patch size.
    :param dilation: Integer of dilation of patch size.
    :return: Patches of images array.
    """

    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out
        
    channels = x.shape[-3]
    h_dim_in = x.shape[-2]
    w_dim_in = x.shape[-1]
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B, C, H, W)
    x = torch.nn.functional.unfold(x, 
                                   kernel_size, 
                                   padding=padding, 
                                   stride=stride, 
                                   dilation=dilation)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_out * w_dim_out)
    x = x.view(-1, channels, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)
    x = x.permute(0,1,4,5,2,3)
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    return x


def combine_patches_2d(x, output_shape, kernel_size, padding=0, stride=1, dilation=1):
    """
    Re-extracting patches from 2D images.
    :param kernel_size: Integer of window of patch size.
    :param padding: Integer of padding of patch size.
    :param stried: Integer of stried of patch size.
    :param dilation: Integer of dilation of patch size.
    :return: Patches of images array.
    """
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 
                   2 * dim_padding - 
                   dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    h_dim_out = output_shape[-2]
    w_dim_out = output_shape[-1]
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B * h_dim_in * w_dim_in, C, kernel_size[0], kernel_size[1])
    x = x.view(-1, channels, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    x = x.permute(0,1,4,5,2,3)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_in, w_dim_in)
    x = x.contiguous().view(-1, 
                            channels * kernel_size[0] * kernel_size[1], 
                            h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    x = torch.nn.functional.fold(x, 
                                 output_size=(h_dim_out, w_dim_out), 
                                 kernel_size=(kernel_size[0], kernel_size[1]), 
                                 padding=padding, 
                                 stride=stride, 
                                 dilation=dilation)
    # (B, C, H, W)
    return x

def extract_patches(array, k, p, s, d):
    """
    Applying extraction of patches from 2D images.
    :param k: Integer of window of patch size.
    :param p: Integer of padding of patch size.
    :param s: Integer of stried of patch size.
    :param d: Integer of dilation of patch size.
    :return: Patches of images array.
    """

    padding = (
        ((array.shape[3]//k+1)*k-array.shape[3])//2, 
        ((array.shape[3]//k+1)*k-array.shape[3])//2,
        ((array.shape[2]//k+1)*k-array.shape[2])//2,
        ((array.shape[2]//k+1)*k-array.shape[2])//2,
    )

    # Input with padding with outer most values
    a = F.pad(array.view(1, -1, array.shape[2], array.shape[3]), padding, "replicate")

    # Patches
    return extract_patches_2d(a, k, padding=p, stride=s, dilation=d)

def combine_patches(patches, shape, k, p, s, d):
    """
    Applying re-extraction of patches from 2D images.
    :param k: Integer of window of patch size.
    :param p: Integer of padding of patch size.
    :param s: Integer of stried of patch size.
    :param d: Integer of dilation of patch size.
    :return: Patches of images array.
    """
    
    padding = (
        ((shape[1]//k+1)*k-shape[1])//2, 
        ((shape[1]//k+1)*k-shape[1])//2,
        ((shape[0]//k+1)*k-shape[0])//2,
        ((shape[0]//k+1)*k-shape[0])//2,
    )
    
    # Combining patches with summation on overlapping pixels
    c = combine_patches_2d(
        patches, 
        (1, 1, (shape[0]//k+1)*k, (shape[1]//k+1)*k), 
        kernel_size=k, padding=p, stride=s, dilation=d)

    # Retrieve back the original image by dividing the unnecessary overlapping summation
    e = torch.ones_like(patches)
    f = combine_patches_2d(
        e, 
        (1, 1, (shape[0]//k+1)*k, (shape[1]//k+1)*k), 
        kernel_size=k, padding=p, stride=s, dilation=d)
    g = c/f
    
    return g[0,:,padding[2]:(padding[2]+shape[0]),padding[0]:(padding[0]+shape[1])]