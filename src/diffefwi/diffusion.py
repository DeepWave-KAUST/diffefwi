"""
Classes to construct the denoising diffusion probabilistic model.
Most are extracted from https://github.com/LinXueyuanStdio/PyTorch-DDPM
@hatsyim
"""

import math
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision

from skimage.transform import resize
from tqdm import tqdm
from abc import abstractmethod
from diffefwi.utils import *
from diffefwi.plots import *
from diffefwi.fwi import fwi_loop

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        :param x: PyTorch module.
        :param emb: timestep embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        :param x: PyTorch module.
        :param emb: timestep embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class ResidualBlock(TimestepBlock):
    """
    A sequential module that constructs a resiudal block
    for a given timestep block.
    """
    
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        Create residual blocks.
        :param x: tensor with dimension [batch_size, in_dim, height, width]
        :param t: tensor with dimension [batch_size, time_dim]
        :return: PyTorch module.
        """
        
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    A sequential module that constructs an attention block.
    """
    
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Create residual blocks.
        :param x: tensor with dimension [batch_size, channels, height, width]
        :return: PyTorch module.
        """
        
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

class UpsampleBlock(nn.Module):
    """
    A sequential module that constructs the upsampling block
    of the U-Net.
    """
    
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class DownsampleBlock(nn.Module):
    """
    A sequential module that constructs the downsampling block
    of the U-Net.
    """
    
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)
    
class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(DownsampleBlock(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(UpsampleBlock(ch, conv_resample))
                    ds //= 2

                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        hs = []
        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        return self.out(h)

class DenoisingDiffusionProbabilisticModel:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear',
        patch=False,
        stride=[1,1],
        kernel_size=[64,64]
    ):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.patch = patch
        self.stride = stride
        self.kernel_size = kernel_size
    
    def _extract(self, a, t, x_shape):
        """
        Get the param of given timestep t.
        """
        
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion (using the nice property): q(x_t | x_0).
        
        :param x_start: x_{i-1} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param noise: specified noise distribution.
        
        :return: x_{i} an [N x C x H x W] Tensor of inputs.
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(self, x_start, t):
        """
        Get the mean and variance of q(x_t | x_0).
        
        :param x_start: x_{i-1} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        
        :return: mean, var, and logvar of x_{i-1}.
        """
        
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Get the mean and variance of the posterior q(x_{t-1} | x_t, x_0).
        
        :param x_start: x_{i-1} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        
        :return: mean, var, and logvar of x_{i-1}.
        """
        
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Compute x_0 from x_t and pred noise: the reverse of `q_sample`.
        
        :param x_start: x_{i-1} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param noise: specified noise distribution.  
        
        :return: x_0.
        """
        
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        """
        Compute predicted mean and variance of p(x_{t-1} | x_t).
        
        :param x_start: x_{i-1} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param noise: specified noise distribution.   
        
        :return: x_0.
        """

        if self.patch:
            pass
            img_patches = split_data_to_size(x_t.view(1,-1,290,527), self.kernel_size, self.stride).to('cuda')
            receiver_locationsecon = torch.empty_like(img_patches)
            for i in range(img_patches.shape[0]):
            
                # predict noise using model
                pred_noise = model(
                    F.interpolate(img_patches[i].unsqueeze(0), size=(256,256)),
                    t
                )
                
                # get the predicted x_0: different from the algorithm2 in the paper
                receiver_locationsecon[i,:,:,:] = F.interpolate(
                    self.predict_start_from_noise(F.interpolate(img_patches[i].unsqueeze(0), size=(256,256)), t, pred_noise), 
                    size=(290,290)
                )
                
            receiver_locationsecon = merge_data_to_size(receiver_locationsecon, [290, 527], self.stride).to('cuda')
            
            if clip_denoised:
                receiver_locationsecon = torch.clamp(receiver_locationsecon, min=-1., max=1.)
            model_mean, posterior_variance, posterior_log_variance = \
                        self.q_posterior_mean_variance(receiver_locationsecon, x_t, t)
        else:
            # Predict noise using model
            pred_noise = model(x_t, t)
    
            # Get the predicted x_0: different from the algorithm2 in the paper
            receiver_locationsecon = self.predict_start_from_noise(x_t, t, pred_noise)
            
            if clip_denoised:
                receiver_locationsecon = torch.clamp(receiver_locationsecon, min=-1., max=1.)
            model_mean, posterior_variance, posterior_log_variance = \
                        self.q_posterior_mean_variance(receiver_locationsecon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        """
        Denoise_step: sample x_{t-1} from x_t and pred_noise.
        
        :param model: PyTorch model.
        :param x_start: x_{i} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param clip_denoised: specified noise distribution.  
        
        :return: x_{i-1}.
        """
        
        # Predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        # Compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_wf(self, model, x_t, t, clip_denoised=True):
        """
        Conditional denoise_step: sample x_{i-1} from x_i and pred_noise.
        
        :param model: PyTorch model.
        :param x_start: x_{i} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param clip_denoised: specified noise distribution.     
        
        :return: x_{i-1} and its mean.
        """
        
        # Predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        # Compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * 0.001
        return pred_img, model_mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape, batches=False):
        """
        Reverse (denoising) diffusion loop.
        
        :param model: PyTorch model.
        :param shape: an [N x C x H x W] array.     
        
        :return: x_{i-1} and its mean.
        """
        
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            img = self.p_sample(
                model, img, 
                torch.full((batch_size,), i, device=device, dtype=torch.long)
            )
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3, batches=False):
        """
        Perform reverse (denoising) diffusion sampling.
        
        :param model: PyTorch model.
        :param image_size: an [H x W] array.      
        :param batch_size: N integer.
        :param channels: C integer.
        
        :return: an [N x C x H x W] images.
        """
        
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), batches=batches)
    
    def train_losses(self, model, x_start, t):
        """
        Compute train losses.
        
        :param model: PyTorch model.
        :param x_start: x_{i} an [N x C x H x W] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        
        :return: loss value.
        """
        
        # Generate random noise
        noise = torch.randn_like(x_start)
        
        # Get the noised version of x_start (x_t)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)

        # Compute loss using an L2 norm
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def fwi_sample(self, model, shape, init_time_step, 
                   vp_img, vs_img, rho_img, 
                   num_epochs, num_shots, dz, dx, dt,
                   receiver_locations, receiver_amplitudes_true,
                   source_locations, source_amplitudes, 
                   freq, data_weight, loss_type='l2', filter=None,
                   learning_rate=20, optim='adam', diffusion=True, 
                   save_dir=None, use_wandb=False, 
                   maxs=[5000,2887,2607], mins=[3000,1732,2294], 
                   num_batches=2, synthetic_mask=None, 
                   elastic=True, grad_norm=False, 
                   grad_smooth=None, use_scheduler=False, inv_source=False, 
                   source_illum=None, grad_clip=1., diffusion_only=False, 
                   regularization=None, regularization_weight=0.0):
        """
        Perform data-fidelity guided (from EFWI) posterior diffusion sampling.
        
        :param model: PyTorch model.
        :param shape: an [N x C x H x W] array.
        :param init_time_step: Initial time step for sampling.
        :param vp_img, vs_img, rho_img: Initial compressional, shear velocity
        and density tensors.
        :param dz, dx, dt: Time and space grid size.
        :param receiver_locations, receiver_amplitudes_true:
        receiver locations and observed data.
        :param source_locations, source_amplitudes:
        Source locations and the source wavelet.
        :param freq: Dominant source frequency (Hz).
        :param data_weight: A list of float of data mismatch weight.
        :param loss_type: Objective function.
        :param filter: Boolean whether a data filter is used during EFWI.
        :param learning_rate: Float for the optimization algorithm.
        :param optim: PyTorch's optimization algorithm.
        :param diffusion: Boolean to toggle the diffusion prior.
        :param save_dir: Folder directory.
        :param wandb: Boolean to indiciate whether wandb is used.
        :param maxs: List of maximum elastic (vp,vs,rho) values.
        :param mins: List of minimum elastic (vp,vs,rho) values.
        
        :return: vp, vs, and rho images and the efwi loss list.
        """
        
        batch_size = shape[0]
        device = next(model.parameters()).device
        vp_imgs = []
        vs_imgs = []
        rho_imgs = []
        loss_fwi_list = []
        for i in tqdm(reversed(range(0, self.timesteps-init_time_step)),
                      desc='Data-fidelity Sampling.', total=self.timesteps-init_time_step):
            if diffusion:
                img = torch.cat((vp_img, vs_img, rho_img), 1)
                if self.patch:
                    _, img_mean = self.p_sample_wf(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
                else:    
                    _, img_mean = self.p_sample_wf(
                        model, 
                        torch.from_numpy(resize(img.detach().cpu().numpy(), (img.shape[0],img.shape[1],256,256))).to(device), 
                        torch.full((batch_size,), i, device=device, dtype=torch.long)
                    )
                img_mean =  torch.from_numpy(resize(img_mean.detach().cpu().numpy(), (img.shape[0],img.shape[1],vp_img.shape[2],vp_img.shape[3]))).to(device)
                
                if diffusion_only==False:
                    print(f'EFWI with Diffusion Priors.')
                    if i != 0:
                        vp_img, vs_img, rho_img, loss_fwi, data_loss = fwi_loop(
                            denormalize_vp(img_mean[:,0,:,:].squeeze(0).squeeze(0), vmin=mins[0], vmax=maxs[0]), 
                            denormalize_vs(img_mean[:,1,:,:].squeeze(0).squeeze(0), vmin=mins[1], vmax=maxs[1]), 
                            denormalize_rho(img_mean[:,2,:,:].squeeze(0).squeeze(0), vmin=mins[2], vmax=maxs[2]), 
                            source_locations, receiver_locations, 
                            dz, dx, dt, num_shots,
                            source_amplitudes, receiver_amplitudes_true,
                            freq, num_epochs, data_weight, save_dir=save_dir, maxs=maxs, mins=mins,
                            num_batches=num_batches, synthetic_mask=synthetic_mask, loss_type=loss_type, 
                            elastic=elastic, grad_norm=grad_norm, use_scheduler=use_scheduler,
                            grad_smooth=grad_smooth,inv_source=inv_source,source_illum=source_illum,
                            grad_clip=grad_clip, regularization=regularization, regularization_weight=regularization_weight
                    )
                else:
                    print(f'Diffusion Priors Sampling.')
                    vp_img, vs_img, rho_img = img_mean[:,0,:,:].unsqueeze(0), img_mean[:,1,:,:].unsqueeze(0), img_mean[:,2,:,:].unsqueeze(0)
                    results = {
                        'vp_inve'   :denormalize_vp(img_mean[:,0,:,:].squeeze(0).squeeze(0), vmin=mins[0], vmax=maxs[0]).detach().cpu().numpy(),
                        'vs_inve'   :denormalize_vs(img_mean[:,1,:,:].squeeze(0).squeeze(0), vmin=mins[1], vmax=maxs[1]).detach().cpu().numpy(),
                        'rho_inve'  :denormalize_rho(img_mean[:,2,:,:].squeeze(0).squeeze(0), vmin=mins[2], vmax=maxs[2]).detach().cpu().numpy()
                    }
                    plot_moduli(
                        denormalize_vp(img_mean[:,0,:,:].squeeze(0).squeeze(0), vmin=mins[0], vmax=maxs[0]).detach().cpu().numpy(),
                        denormalize_vs(img_mean[:,1,:,:].squeeze(0).squeeze(0), vmin=mins[1], vmax=maxs[1]).detach().cpu().numpy(),
                        denormalize_rho(img_mean[:,2,:,:].squeeze(0).squeeze(0), vmin=mins[2], vmax=maxs[2]).detach().cpu().numpy(),
                        'moduli_inve'+'_'+str(int(time.time()))+'.pdf', save_dir
                    )
                    torch.save(results,save_dir+'/inverted'+str(i)+'_'+str(int(time.time()))+'.tz')
            else:
                print(f'EFWI without Diffusion Priors.')
                img = torch.cat((vp_img, vs_img, rho_img), 1)
                vp_img, vs_img, rho_img, loss_fwi, data_loss = fwi_loop(
                    denormalize_vp(img[:,0,:,:].squeeze(0).squeeze(0), vmin=mins[0], vmax=maxs[0]), 
                    denormalize_vs(img[:,1,:,:].squeeze(0).squeeze(0), vmin=mins[1], vmax=maxs[1]), 
                    denormalize_rho(img[:,2,:,:].squeeze(0).squeeze(0), vmin=mins[2], vmax=maxs[2]), 
                    source_locations, receiver_locations, 
                    dz, dx, dt, num_shots, 
                    source_amplitudes, receiver_amplitudes_true,
                    freq, num_epochs, data_weight, save_dir=save_dir, maxs=maxs, mins=mins,
                    num_batches=num_batches, synthetic_mask=synthetic_mask, loss_type=loss_type, 
                    elastic=elastic, grad_norm=grad_norm, use_scheduler=use_scheduler,
                    grad_smooth=grad_smooth,inv_source=inv_source,source_illum=source_illum,
                    grad_clip=grad_clip, regularization=regularization, regularization_weight=regularization_weight
                )
                
            vp_img = torch.clamp(vp_img, min=-1., max=1.)
            vs_img = torch.clamp(vs_img, min=-1., max=1.)
            rho_img = torch.clamp(rho_img, min=-1., max=1.)

            if (use_wandb==True) and (diffusion_only==False):
                wandb.log({"fwi_loss": loss_fwi})
                wandb.log({"data_loss": data_loss})
            
            vp_imgs.append(vp_img.cpu().numpy())
            vs_imgs.append(vs_img.cpu().numpy())
            rho_imgs.append(rho_img.cpu().numpy())
            if diffusion_only==False:
                loss_fwi_list.append(loss_fwi)
        return vp_imgs, vs_imgs, rho_imgs, loss_fwi_list