import torch
import deepwave
import time
import torchvision

# Debugging NaNs
torch.autograd.set_detect_anomaly(True)

from diffefwi.plots import *
from diffefwi.regularization import *
from diffefwi.utils import *
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter
from scipy.ndimage import gaussian_filter

def source_illumination(vp_smooth, vs_smooth, rho_smooth, 
                        dz, dx, dt, batch_src_amps, batch_source_locations, 
                        batch_receiver_locations):
    """
    Compute the graph space optimal transport objective function.

    :param y_pred: The predicted signal, 
    of dimensions [shot, receiver, time]
    :param y: The true signal
    :param eta: Parameter controlling the relative weight 
    of the two terms in the cost function (which relate 
    to differences in the time and amplitude dimensions)

    :return: The cost function value/loss.
    """
    
    nz, nx = model.shape
    ns, _, nt = source.shape
    num_batches = ns
    result = torch.zeros((1, nz*nx, nt))
    for it in tqdm(range(num_batches)):
        source_wavefield = deepwave.elastic(
            *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_smooth, vs_smooth, rho_smooth),
            [dz,dx], dt, source_amplitudes_y=batch_src_amps,
            source_locations_y=batch_source_locations,
            receiver_locations_y=batch_receiver_locations,
            pml_width = [0, 50, 50, 50],
            accuracy = 4,
        )[-2]
        result += source_wavefield.detach().cpu().clone() ** 2

    src_illum = result.reshape(nx, nz, nt)
    src_illum = torch.sum(src_illum, dim=-1)
    return src_illum.T

def optimal_transport(y_pred, y, eta, device):
    """
    Compute the graph space optimal transport objective function.

    :param y_pred: The predicted signal, 
    of dimensions [shot, receiver, time]
    :param y: The true signal
    :param eta: Parameter controlling the relative weight 
    of the two terms in the cost function (which relate 
    to differences in the time and amplitude dimensions)

    :return: The cost function value/loss.
    """
    loss = torch.tensor(0, dtype=torch.float).to(device)
    for s in range(y.shape[0]):
        for r in range(y.shape[1]):
            nt = y.shape[-1]
            c = np.zeros([nt, nt])
            for i in range(nt):
                for j in range(nt):
                    c[i, j] = (
                        eta * (i-j)**2 +
                        (y_pred.detach()[s, r, i]-y[s, r, j])**2
                    )
            row_ind, col_ind = linear_sum_assignment(c)
            y_sigma = y[s, r, col_ind]
            loss = (
                loss + (
                    eta * torch.tensor(row_ind-col_ind).to(device)**2 +
                    (y_pred[s, r]-y_sigma)**2
                ).sum()
            )
    return loss

def cross_correlation(x, y):
    """
    Perform cross-correlation objective function.
    
    :param x: A tensor of synthetic data.
    
    :return: Cross-correlation between x and y.
    """
    
    x = x/torch.norm(x)
    y = y/torch.norm(y)
    loss = -torch.sum(torch.mul(x,y))
    return loss

def fwi_loop(
    vp, vs, rho, 
    source_locations, receiver_locations, dz, dx, dt, num_shots, 
    source_amplitudes, receiver_amplitudes_true, freq,
    num_epochs, data_weight=[1.,1.], loss_type='l2', filter=None,
    learning_rate=20, optim='adam', save_dir=None, 
    maxs=[5000,2887,2607], mins=[3000,1732,2294], num_batches=2,
    synthetic_mask=None, elastic=True, gradient=False, 
    grad_norm=False, grad_smooth=False, use_scheduler=False, 
    inv_source=False, source_illum=None, grad_clip=1.,
    regularization=None, regularization_weight=0.0):
    """
    Perform data-fidelity guided (from EFWI) posterior diffusion sampling.
    
    :param vp_img, vs_img, rho_img: Initial compressional, shear velocity
    and density tensors.
    :param receiver_locations, receiver_amplitudes_true:
    receiver locations and observed data.
    :param dx, dt: Time and space grid size.
    :param source_locations, source_amplitudes: 
    Source locations and the source wavelet.
    :param learning_rate: Float for the optimization algorithm.
    :param optim: PyTorch's optimization algorithm.
    :param maxs: List of maximum elastic (vp,vs,rho) values.
    :param mins: List of minimum elastic (vp,vs,rho) values.
    
    :return: vp, vs, and rho images and the efwi (weighted data), model, data loss list.
    """

    device = vp.device
    vp_smooth = vp.clone().detach().requires_grad_(True)
    vs_smooth = vs.clone().detach().requires_grad_(True)
    rho_smooth = rho.clone().detach().requires_grad_(True)
    
    if optim=='adam':
        optimizer = torch.optim.Adam([{'params': [vp_smooth, vs_smooth, rho_smooth], 'lr': learning_rate}])
    else:
        optimizer = torch.optim.LBFGS([vp_smooth, vs_smooth, rho_smooth])

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_epochs + 1, 0
        )

    if inv_source:
        source_amplitudes.requires_grad=True
        optimizer_source = torch.optim.Adam([{'params':[source_amplitudes], 'lr': learning_rate}])
    
    for epoch in range(num_epochs):

        # print(epoch)
        
        epoch_loss = 0.0
        diff_loss = 0.0
        data_loss = 0.0
        optimizer.zero_grad()
        if inv_source:  
            optimizer_source.zero_grad()

        # Bounds projection and smoothing
        vp_smooth.data[vp_smooth.data < mins[0]] = mins[0]
        vp_smooth.data[vp_smooth.data > maxs[0]] = maxs[0]
        vs_smooth.data[vs_smooth.data < mins[1]] = mins[1]
        vs_smooth.data[vs_smooth.data > maxs[1]] = maxs[1]
        rho_smooth.data[rho_smooth.data < mins[2]] = mins[2]
        rho_smooth.data[rho_smooth.data > maxs[2]] = maxs[2]
        vp_smooth.data = (
            torchvision.transforms.functional.gaussian_blur(
                vp_smooth[None], [3,3]
            ).squeeze()
        )
        vs_smooth.data = (
            torchvision.transforms.functional.gaussian_blur(
                vs_smooth[None], [3,3]
            ).squeeze()
        )
        rho_smooth.data = (
            torchvision.transforms.functional.gaussian_blur(
                rho_smooth[None], [3,3]
            ).squeeze()
        )

        # Perform loss calculation per batch
        for it in range(0, num_shots, num_batches):
            batch_src_amps = source_amplitudes[it:it+num_batches, :, :]
            batch_rcv_amps_true = receiver_amplitudes_true[it:it+num_batches, :, :].to(device)
            batch_source_locations = source_locations[it:it+num_batches, :, :].to(device)
            batch_receiver_locations = receiver_locations[it:it+num_batches, :, :].to(device)
            if elastic:
                batch_rcv_amps_pred = deepwave.elastic(
                    *deepwave.common.vpvsrho_to_lambmubuoyancy(vp_smooth, vs_smooth, rho_smooth),
                    [dz,dx], dt, source_amplitudes_y=batch_src_amps,
                    source_locations_y=batch_source_locations,
                    receiver_locations_y=batch_receiver_locations,
                    accuracy=4,
                    pml_freq=freq,
                    pml_width=[50, 50, 50, 50]
                )[-2]#[:,:,::2]
            else:
                batch_rcv_amps_pred = deepwave.scalar(
                    vp_smooth, 
                    dt=dt, 
                    grid_spacing=[dz,dx],
                    source_amplitudes=batch_src_amps,
                    source_locations=batch_source_locations,
                    receiver_locations=batch_receiver_locations,
                    accuracy=8,
                    # pml_freq=freq,
                    pml_width=[50, 50, 50, 50]
                )[-1]
            if synthetic_mask is not None:
                mask = [
                    synthetic_mask[0][it:it+num_batches, :, :].to(device), 
                    synthetic_mask[1][it:it+num_batches, :, :].to(device)
                ]
            else:
                mask = [torch.ones_like(batch_rcv_amps_pred),torch.ones_like(batch_rcv_amps_pred)]

            data_loss = 0.0
            if filter is not None:
                # Filter data
                for cutoff_freq in [filter]:
                    sos = butter(6, cutoff_freq, fs=1/dt, btype='highpass', output='sos')
                    sos = [torch.tensor(sosi).to(batch_rcv_amps_true.dtype).to(device) for sosi in sos]
                    def filt(x):
                        return biquad(biquad(biquad(x, *sos[0]), *sos[1]), *sos[2])
                    batch_rcv_amps_pred_ = filt(batch_rcv_amps_pred)
                    batch_rcv_amps_true_ = filt(batch_rcv_amps_true)
                    
            # # Data scaling
            # batch_rcv_amps_pred *= (mask[0] + 1e-16)
            # batch_rcv_amps_pred = torch.divide(
            #     batch_rcv_amps_pred, 
            #     (torch.max(batch_rcv_amps_pred, axis=-1)[0]
            #      .view(batch_rcv_amps_pred.shape[0],-1,1)
            #      .repeat(1,1,batch_rcv_amps_pred.shape[-1])
            #     )
            # )

            if loss_type=='l2':
                data_loss += torch.nn.MSELoss()(
                    data_weight[0]*batch_rcv_amps_pred*mask[0] + 1e-6, 
                    data_weight[1]*batch_rcv_amps_true*mask[1] + 1e-6
                )
            elif loss_type=='xc':
                data_loss += cross_correlation(
                    data_weight[0]*batch_rcv_amps_pred*mask[0] + 1e-6, 
                    data_weight[1]*batch_rcv_amps_true*mask[1] + 1e-6
                )
            elif loss_type == 'gsot':
                data_loss += optimal_transport(
                    data_weight[0]*batch_rcv_amps_pred*mask[0] + 1e-6, 
                    data_weight[1]*batch_rcv_amps_true*mask[1] + 1e-6, 
                    eta=0.003, device=device
                )
            if (regularization == 'None') or regularization is None:
                reg = 0.0
            elif regularization == 'l2':
                reg =  regularization_weight * (
                    l2_regularization(vp_smooth) +
                    l2_regularization(vs_smooth) + 
                    l2_regularization(rho_smooth)
                )
            elif regularization == 'tv_l2':
                reg = regularization_weight * (
                    tv_l2_regularization(vp_smooth) +
                    tv_l2_regularization(vs_smooth) + 
                    tv_l2_regularization(rho_smooth)
                )
            elif regularization == 'tv_l1':
                reg =  regularization_weight * (
                    tv_l1_regularization(vp_smooth) +
                    tv_l1_regularization(vs_smooth) + 
                    tv_l1_regularization(rho_smooth)
                )
            elif regularization == 'tikhonov_2nd':
                reg = regularization_weight * (
                    second_order_tikhonov_regularization(vp_smooth) +
                    second_order_tikhonov_regularization(vs_smooth) + 
                    second_order_tikhonov_regularization(rho_smooth)
                )
            elif regularization == 'tikhonov_1st':
                reg = loss = data_loss + regularization_weight * (
                    tikhonov_regularization(vp_smooth) +
                    tikhonov_regularization(vs_smooth) + 
                    tikhonov_regularization(rho_smooth)
                )
            loss = data_loss + reg    
            epoch_loss += loss.item()
            data_loss += data_loss.item()
            loss.backward(retain_graph=True)

            # Save data for plotting: the middle shot
            plot_id=0 #12
            if it==0: #120
                if epoch==0:
                    plot_init = data_weight[0]*batch_rcv_amps_pred[plot_id]*mask[0][plot_id]
                plot_dsyn = data_weight[0]*batch_rcv_amps_pred[plot_id]*mask[0][plot_id]
                plot_dobs = data_weight[1]*batch_rcv_amps_true[plot_id]*mask[1][plot_id]
                mask_dsyn = mask[0][plot_id]
                mask_dobs = mask[1][plot_id]
            
        # Check suspicious gradient updates
        if (loss.isnan().sum()>0):
            print('Discovered NaNs in the data.')
            
        # Normalize the gradient
        if epoch == 0:
            vp_gmax = torch.max(abs(vp_smooth.grad))
            vs_gmax = torch.max(abs(vs_smooth.grad))
            rho_gmax = torch.max(abs(rho_smooth.grad))

        if grad_norm:
            if inv_source: 
                source_gmax = torch.max(abs(source_amplitudes.grad))
                source_amplitudes.grad = source_amplitudes.grad/source_gmax         
    
            vp_smooth.grad /= vp_gmax
            vs_smooth.grad /= vs_gmax
            rho_smooth.grad /= rho_gmax

        # Source illumination weights
        if source_illum is not None:
            vp_smooth.grad *= source_illum
            vs_smooth.grad *= source_illum
            rho_smooth.grad *= source_illum
            
        # Smooth gradient
        if grad_smooth is not None:
            vp_smooth.grad = torch.tensor(gaussian_filter(vp_smooth.grad.cpu().numpy(), [grad_smooth, grad_smooth])).to(device)
            vs_smooth.grad = torch.tensor(gaussian_filter(vs_smooth.grad.cpu().numpy(), [grad_smooth, grad_smooth])).to(device)
            rho_smooth.grad = torch.tensor(gaussian_filter(rho_smooth.grad.cpu().numpy(), [grad_smooth, grad_smooth])).to(device)
            
            vp_gmax = torch.max(abs(vp_smooth.grad))
            vs_gmax = torch.max(abs(vs_smooth.grad))
            rho_gmax = torch.max(abs(rho_smooth.grad))

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(vp_smooth, grad_clip*vp_gmax)
            torch.nn.utils.clip_grad_norm_(vs_smooth, grad_clip*vs_gmax)
            torch.nn.utils.clip_grad_norm_(rho_smooth, grad_clip*rho_gmax)

        optimizer.step()
        if inv_source:
            optimizer_source.step()

        if use_scheduler:
            scheduler.step()

        # Check suspicious gradient updates
        if ((vp_smooth.isnan().sum()>0) or 
            (vs_smooth.isnan().sum()>0) or 
            (rho_smooth.isnan().sum()>0)):
            print('Loss: '+str(loss))
            print('Discovered NaNs in the model.')
        
        if epoch==(num_epochs-1):  
            print('Loss: '+str(data_loss))
            print('Regularization: '+str(reg))
            plot_gather(
                plot_dobs, 
                plot_dsyn, 
                'gather'+'_'+str(int(time.time()))+'.pdf', save_dir,
                cmap='gray'
            )
            plot_gather(
                plot_dsyn-plot_init, 
                plot_dsyn-plot_dobs, 
                'residual'+'_'+str(int(time.time()))+'.pdf', save_dir,
                cmap='gray'
            )
            plot_gather(
                mask_dobs, 
                mask_dsyn, 
                'mask'+'_'+str(int(time.time()))+'.pdf', save_dir,
                cmap='seismic'
            )
            plot_moduli(
                vp_smooth.detach().cpu().numpy(), 
                vs_smooth.detach().cpu().numpy(), 
                rho_smooth.detach().cpu().numpy(), 
                'moduli_inve'+'_'+str(int(time.time()))+'.pdf', save_dir
            )
            plot_moduli(
                vp_smooth.grad.detach().cpu().numpy(), 
                vs_smooth.grad.detach().cpu().numpy(), 
                rho_smooth.grad.detach().cpu().numpy(), 
                'moduli_grad'+'_'+str(int(time.time()))+'.pdf', save_dir
            )
            plot_moduli(
                vp_smooth.detach().cpu().numpy()/vs_smooth.detach().cpu().numpy(), 
                vp_smooth.detach().cpu().numpy()/rho_smooth.detach().cpu().numpy(), 
                vs_smooth.detach().cpu().numpy()/rho_smooth.detach().cpu().numpy(), 
                'ratio_inve'+'_'+str(int(time.time()))+'.pdf', save_dir)

            results = {
                'vp_inve'   :vp_smooth.detach().cpu().numpy(),
                'vs_inve'   :vs_smooth.detach().cpu().numpy(),
                'rho_inve'  :rho_smooth.detach().cpu().numpy()
            }
            torch.save(results,save_dir+'/inverted'+str(epoch)+'_'+str(int(time.time()))+'.tz')

    if gradient:
        return (normalize_vp(vp_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[0], vmax=maxs[0]), 
                normalize_vs(vs_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[1], vmax=maxs[1]), 
                normalize_rho(rho_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[2], vmax=maxs[2]), 
                epoch_loss, data_loss)
    else:
        return (normalize_vp(vp_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[0], vmax=maxs[0]), 
                normalize_vs(vs_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[1], vmax=maxs[1]), 
                normalize_rho(rho_smooth.detach().unsqueeze(0).unsqueeze(0), vmin=mins[2], vmax=maxs[2]), 
                epoch_loss, data_loss)