import h5py
import sys

from diffefwi.diffusion import *
from diffefwi.plots import *
from diffefwi.utils import *
from pathlib import Path

from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser

plt.style.use("../hatsyim.mplstyle")

def main():
    
    # Saving directory
    save_dir = '../saves/diffusion/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data = np.load('../data/06_overthrust_c128_s128_128.npy')
    
    from torch.utils.data import TensorDataset, DataLoader
    
    batch_size = 16
    timesteps = 1000
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data).float()), batch_size=batch_size, shuffle=True)
        
    # Define model and diffusion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3
    ).to(device)
    model.to(device)
    model.load_state_dict(torch.load('../saves/diffusion/combined.pt'))
    gaussian_diffusion = DenoisingDiffusionProbabilisticModel(timesteps=timesteps)

    # Generate new images
    generated_images = gaussian_diffusion.sample(model, 256, batch_size=16, channels=3)
    imgs = generated_images[-1].reshape(4, 4, 3, 256, 256)
    
    vp_gen = imgs[:,:,0,:,:]
    vs_gen = imgs[:,:,1,:,:]
    rho_gen = imgs[:,:,2,:,:]
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(vp_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_vp_combined.pdf')
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(vs_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_vs_combined.pdf')
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(rho_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_rho_combined.pdf')
    
    vp_gen = imgs[:,:,0,:,:]/imgs[:,:,1,:,:]
    vs_gen = imgs[:,:,0,:,:]/imgs[:,:,2,:,:]
    rho_gen = imgs[:,:,1,:,:]/imgs[:,:,2,:,:]
    
    print('Generated Vp')
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(vp_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_vpvs_combined.pdf')
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(vs_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_vprho_combined.pdf')
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for n_row in range(4):
        for n_col in range(4):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            f_ax.imshow(rho_gen[n_row, n_col])
            f_ax.axis("off")
    plt.savefig(save_dir+'/gen_diff_vsrho_combined.pdf')
    
if __name__ == '__main__':
    main()