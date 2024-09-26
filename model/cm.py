"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import torch as th
import torch.nn as nn
import numpy as np
import copy
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 feature_dim=50,
                 hidden_dim=1024,
                 t_dim=16,
                 ln=True):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            # nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim

        # hidden_dim = 1024
        # feature_dim = 512


        if ln:

            self.mid_layer = nn.Sequential(nn.Linear(input_dim, feature_dim),
                                        nn.LayerNorm(feature_dim),
                                        nn.Tanh(),
                                        # nn.Mish(),
                                        # nn.ReLU(),
                                        nn.Linear(feature_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        # nn.LayerNorm(hidden_dim),
                                        # nn.Mish())
                                        nn.ReLU(inplace=True))
        else:
            self.mid_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish(),
                                        #    nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Mish())

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):

        t = self.time_mlp(time)

        x = th.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    else:
        raise NotImplementedError

class DummyGenerator:
    def randn(self, *args, **kwargs):
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return th.randn_like(*args, **kwargs)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return th.cat([x, x.new_zeros([1])])

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class ConsistencyModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        feature_dim,
        hidden_dim,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        # ts=(13,5,19,19,32),
        sample_steps=2,
        generator=None,
        sampler="onestep", 
        clip_denoised=True,
        ln=False,
    ):
        super(ConsistencyModel, self).__init__()
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho

        self.device = device

        if generator is None:
            self.generator = get_generator("dummy")
        else:
            self.generator = generator

        self.sampler = sampler
        self.steps = steps
        self.ts = [i for i in range(0, steps, sample_steps)]

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, 
                         device=device, ln=ln, 
                         feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
        # self.model = MLP_v1(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)
        # self.model = FiLM(state_dim=state_dim, action_dim=action_dim, device=device, ln=ln).to(device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)
    
    def consistency_losses(
        self,
        x_start,
        state,
        # num_scales=40,
        noise=None,
        target_model=None,
    ):
        num_scales = self.steps

        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            target_model = self.model
        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(self.model, x, t, state)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state)

        x_t2 = euler_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # snr低时，weights 也比较低

        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        return consistency_loss.mean()
    
    def loss(self, x_start, state, noise=None, td_weights=None):
        num_scales = self.steps
        if noise is None:
            noise = th.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(self.model, x, t, state)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t, state)
        recon_diffs = (distiller - x_start) ** 2

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        recon_loss = mean_flat(recon_diffs) * weights

        if td_weights is not None:
            td_weights = th.squeeze(td_weights)
            recon_loss = recon_loss * td_weights
        return recon_loss.mean()
    
    def denoise(self, model, x_t, sigmas, state, return_dict=False):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # rescaled_t = sigmas
        if return_dict:
            model_output, neurons_percent = model(c_in * x_t, rescaled_t, state, return_dict)
        else:
            model_output = model(c_in * x_t, rescaled_t, state)
        denoised = c_out * model_output + c_skip * x_t
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        if return_dict:
            return model_output, denoised, neurons_percent
        else:
            return model_output, denoised

    def sample(self, state, eval=False):
        if self.sampler == "onestep":  
            x_0 = self.sample_onestep(state, eval=eval)
        elif self.sampler == "multistep":
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")

        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)

        return x_0
    
    def sample_onestep(self, state, eval=False, return_dict=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        if return_dict:
            _, denoised, neurons_percent = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state, return_dict=return_dict)
            return denoised, neurons_percent
        else:
            return self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
    
    def sample_multistep(self, state, eval=False):
        x_T = self.generator.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        # x = self.denoise(self.model, x_T, self.sigmas[0] * s_in, state)[1]
        x = x_T
        for i in range(len(self.ts)-1):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            x0 = self.denoise(self.model, x, t * s_in, state)[1]
            next_t = (t_max_rho + self.ts[i+1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            next_t = np.clip(next_t, self.sigma_min, self.sigma_max)
            x = x0 + self.generator.randn_like(x) * np.sqrt(next_t**2 - self.sigma_min**2)
        
        t = (t_max_rho + self.ts[-1] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
        x = self.denoise(self.model, x, t * s_in, state)[1]

        return x
    
    def forward(self, state, eval=False, multistep=False, return_dict=False):
        neurons_percent = dict()
        if multistep:
            x_0 = self.sample_multistep(state, eval=eval)
        else:
            if return_dict:
                x_0, neurons_percent = self.sample_onestep(state, eval=eval, return_dict=return_dict)
            else:
                x_0 = self.sample_onestep(state, eval=eval)
        if self.clip_denoised:
            x_0 = x_0.clamp(-1, 1)
        if return_dict:
            return x_0, neurons_percent
        else:
            return x_0