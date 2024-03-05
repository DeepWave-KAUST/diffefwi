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
        "--loss_type",
        type=str,
        default='l2',
        help="Type of objective function.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default='adam',
        help="Type of optimization algorithm.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default='Diffusion EFWI training.',
        help="Project desciprtion.",
    )
    parser.add_argument(
        "--density_scaler",
        type=float,
        default=1e3,
        help="Scaling for meidum density.",
    )
    parser.add_argument(
        "--data_weight",
        type=float,
        default=1.,
        help="Weighting for the observed data.",
    )
    parser.add_argument(
        "--galat",
        type=float,
        default=800.,
        help="Elastic moduli galat bounds.",
    )
    parser.add_argument(
        "--gaussian_window",
        type=float,
        default=10.,
        help="Window size for Gaussian gradient smoothing.",
    )
    parser.add_argument(
        "--fwi_iteration",
        type=int,
        default=5,
        help="Number of FWI iterations.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.1,
        help="Gradient clipping scaler.",
    )
    parser.add_argument(
        "--vp_max",
        type=float,
        default=2500.,
        help="Maximum compressional velocity in m/s.",
    )

    args = parser.parse_args()
    dict_args = vars(args)
    print(args.description)
    print(dict_args)
    
    # Change these lines for the wandb setup
    save_dir = '../saves/efwi/'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(5637)
    
    # Set parameters
    x_old = np.linspace(0,526,527)
    z_old = np.linspace(0,30,100)
    X_old,Z_old = np.meshgrid(x_old,z_old)
    dz_old = z_old[1]-z_old[0]
    dx_old = x_old[1]-x_old[0]
    x = np.linspace(0,526,527)
    z = np.linspace(0,3*dz_old*290,291)[:-1]
    X,Z = np.meshgrid(x,z)
    dz = z[1]-z[0]
    dx = x[1]-x[0]
    freq = 40
    dz = z[1]-z[0]
    dx = x[1]-x[0]
    dt = 0.001 # 1ms
    fs = 1/dt
    nt = int(0.2 / dt) # 0.5s
    num_dims = 2
    num_shots = 264 #259
    num_sources_per_shot = 1
    num_receivers_per_shot = 264 #256
    source_spacing = 20.0 #30.0
    receiver_spacing = 20.0
    device = torch.device('cuda')
    
    # Smoothed initial model
    vp_init = np.zeros((290*3,527))
    vp_init[:, :527] = np.linspace(1500,args.vp_max,(290*3)).repeat(527).reshape(-1,527)
    vp_init = vp_init[::3,:]
    vp_init = torch.from_numpy(vp_init).to(device).float()
    vs_init = vp_to_vs(vp_init)
    rho_init = vp_to_rho(vp_init)*args.density_scaler
    
    # Add perturbations
    check_vp = torch.from_numpy(1000*(np.sin(.125/4*X)*np.sin(0.25/4*Z)))
    
    # Create elastic checkerboards
    vp_true = vp_init.clone().float().to(device) + check_vp.float().to(device)
    vs_true = vp_to_vs(vp_true)
    rho_true = vp_to_rho(vp_true)*args.density_scaler
    
    plot_moduli(
        vp_true.detach().cpu().numpy(), 
        vs_true.detach().cpu().numpy(), 
        rho_true.detach().cpu().numpy(),
        'moduli_true.pdf', save_dir
    )

    # FWI parameters
    vp = vp_init.clone().float().requires_grad_().to(device)
    vs = vs_init.clone().float().requires_grad_().to(device)
    rho = rho_init.clone().float().requires_grad_().to(device)

    source_locations = torch.zeros(num_shots, num_sources_per_shot, num_dims)
    source_locations[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
    source_locations = source_locations/10
    source_locations[:, 0, 0] += 0
    source_locations = source_locations[:-1]
    
    source_amplitudes_init = (
        deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
        .repeat(num_shots-1, num_sources_per_shot, 1)
        .to(device)
    )
    source_amplitudes = source_amplitudes_init.clone()
    source_amplitudes = source_amplitudes.to(device)

    receiver_locations = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
    receiver_locations[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
    receiver_locations[:, :, 1] = receiver_locations[0, :, 1].repeat(num_shots, 1)
    receiver_locations = receiver_locations/10
    receiver_locations[:, :, 0] += 0
    receiver_locations = receiver_locations[:-1,:-1,:]

    # Synthetic data
    data = deepwave.elastic(
        *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_true, vs_true, rho_true),
        dx, dt, source_amplitudes_y=source_amplitudes.to(device),
        source_locations_y=source_locations.to(device),
        receiver_locations_y=receiver_locations.to(device),
        accuracy=4,
        pml_freq=freq,
        pml_width=[50, 50, 50, 50]
    )[-2]

    # Remove outlier shot #122
    keep_idx = list(range(263))
    keep_idx.pop(122)
    data = data[keep_idx,:,:]
    source_locations = source_locations[keep_idx,:,:]
    receiver_locations = receiver_locations[keep_idx,:,:]
    source_amplitudes = source_amplitudes[keep_idx,:,:]

    # Remove 2 shots: 1 due to outlier and the other due to Deepwave
    num_shots=num_shots-2

    model = UNetModel(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=3
    )
    model.to(device)
    model.load_state_dict(torch.load('../saves/diffusion/combined.pt'))
    timesteps=1000    
    ddpm = DenoisingDiffusionProbabilisticModel(timesteps)

    # Run Diffusion FWI
    vp_diff,vs_diff,rho_diff,loss_fwi_diffusionFWI = ddpm.fwi_sample(
        model, (1, 3, 64, 64),
        900, 
        normalize_vp(
            vp.clone().detach(), vmax=vp_init.max()+args.galat, vmin=vp_init.min()
        ).unsqueeze(0).unsqueeze(0),
        normalize_vs(
            vs.clone().detach(), vmax=vs_init.max()+args.galat, vmin=vs_init.min()             
        ).unsqueeze(0).unsqueeze(0),
        normalize_rho(
            rho.clone().detach(), vmax=rho_init.max()+args.galat, vmin=rho_init.min()  
        ).unsqueeze(0).unsqueeze(0),
        args.fwi_iteration, num_shots, dz, dx, dt,
        receiver_locations, data.float().to(device),
        source_locations, source_amplitudes, 
        freq=freq, data_weight=[args.data_weight,args.data_weight],
        loss_type=args.loss_type, learning_rate=20, optim='adam',save_dir=save_dir,
        maxs=[vp_init.max()+args.galat,vs_init.max()+vp_to_vs(args.galat),rho_init.max()+vp_to_rho(args.galat)], 
        mins=[vp_init.min(),vs_init.min(),rho_init.min()],
        diffusion=True, num_batches=1, use_wandb=False, 
        grad_norm=True, grad_smooth=1,
        inv_source=False, source_illum=None, grad_clip=None
    )
    
    results = {
    'vp_init'   :vp_init.detach().cpu().numpy(),
    'vs_init'   :vs_init.detach().cpu().numpy(),
    'rho_init'  :rho_init.detach().cpu().numpy(),
    'vp_inve'   :vp_diff,
    'vs_inve'   :vs_diff,
    'rho_inve'  :rho_diff,
    'loss'      :loss_fwi_diffusionFWI,
    }
    torch.save(results,save_dir+'/result.tz')
    
if __name__ == "__main__":
    main()