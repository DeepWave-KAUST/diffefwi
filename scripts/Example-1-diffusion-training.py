import h5py
import sys

from diffefwi.diffusion import *
from diffefwi.fwi import *
from diffefwi.plots import *
from diffefwi.utils import *
from pathlib import Path

from pathlib import Path
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader

plt.style.use("../hatsyim.mplstyle")

def main():
    parser = ArgumentParser(description="Diffusion EFWI training.")
    parser.add_argument(
            "--data",
            type=str,
            default='combine',
            help="Type of objective function."
    )

    args = parser.parse_args()

    save_dir = '../saves/diffusion/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # The data provided are subsampled files containing only 256 images
    
    if args.data=='combine':
        print('Using '+args.data)
        data = np.load('../data/combined_64.npy')[:,:,::2,::2] # Reshape 256 to 128 images
    elif args.data=='eopenfwi':
        print('Using '+args.data)
        data = np.load('../data/eopenfwi_64.npy')[:,:,::2,::2] # Reshape 256 to 128 images
    else:
        print('Using '+args.data)
        data = np.load('../data/seg_64.npy')[:,:,::2,::2] # Reshape 256 to 128 images

    batch_size = 10
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
    gaussian_diffusion = DenoisingDiffusionProbabilisticModel(timesteps=timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    
    epochs = 500
    
    last_loss = 1e16
    for epoch in range(epochs):
        total_loss = 0.0
        # for step, (images, labels) in enumerate(train_loader):
        for step, images in enumerate(train_loader):
            optimizer.zero_grad()
    
            batch_size = images[0].shape[0]
            images = images[0].to(device)
    
            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    
            loss = gaussian_diffusion.train_losses(model, images, t)    
            loss.backward()
            optimizer.step()
            total_loss += loss

        if epoch % 1 == 0:
            
            print("Loss:", total_loss)

            if total_loss < last_loss:
                torch.save(model.state_dict(), save_dir+'/model.pt')
                torch.save(optimizer.state_dict(), save_dir+'/optim.pt')

            sample = gaussian_diffusion.sample(model, 256, batch_size=1, channels=3)[-1]
            
            plt.figure()
            plt.imshow(sample[-1,0,:,:])
            plt.savefig(save_dir+'/ddpm-vp_'+str(epoch)+'.pdf')
            plt.figure()
            plt.imshow(sample[-1,1,:,:])
            plt.savefig(save_dir+'/ddpm-vs_'+str(epoch)+'.pdf')
            plt.figure()
            plt.imshow(sample[-1,2,:,:])
            plt.savefig( save_dir+'/ddpm-rho_'+str(epoch)+'.pdf')
   

        last_loss = total_loss

    torch.save(model.state_dict(), save_dir+'model.pt')
    torch.save(optimizer.state_dict(), save_dir+'optim.pt')


if __name__ == "__main__":
    main()