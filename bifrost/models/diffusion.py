import torch
import torch.nn as nn

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
# from diffusers.training_utils import compute_density_for_timestep_sampling



# def compute_density_for_timestep_sampling(
#     weighting_scheme: str,
#     batch_size: int,
#     logit_mean: float = None,
#     logit_std: float = None,
#     mode_scale: float = None,
# ):
#     """
#     weighting_scheme='logit_normal'
#     batch_size=10000
#     logit_mean=0.0
#     logit_std=1.0
#     mode_scale=1.29
#     """
#     u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
#     u = torch.nn.functional.sigmoid(u)
#     return u


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u




# Copied from https://github.com/huggingface/diffusers/blob/5802c2e3f27c6b45290773691bbece4091b69ddc/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L69
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class Diffusion:

    def __init__(self, configs, **kwargs):

        # self.noise_scheduler_type = configs.noise_scheduler_type # if hasattr(configs, 'noise_scheduler_type') else 'FlowMatchEulerDiscreteScheduler'
        # self.noise_scheduler_type = kwargs.get('noise_scheduler_type', 'FlowMatchEulerDiscreteScheduler')
        # if self.noise_scheduler_type == "FlowMatchEulerDiscreteScheduler":
        #     self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        # elif self.noise_scheduler_type == "DDPMScheduler":
        #     # self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, shift=3.0)
        #     self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        

        # self.diffusion_noise_type = configs.diffusion_noise_type # if hasattr(configs, 'diffusion_noise_type') else 'random_gaussian'
        self.vision_denoising_type = configs.vision_denoising_type # if hasattr(configs, 'vision_denoising_type') else 'continuous_diffusion'
        # self.timestep_sampling_strategy = configs.timestep_sampling_strategy # if hasattr(configs, 'timestep_sampling_strategy') else 'logit_normal'
        # self.diffusion_noise_type = kwargs.get('diffusion_noise_type', 'random_gaussian')
        # self.vision_denoising_type = kwargs.get('vision_denoising_type', 'continuous_diffusion')
        # self.timestep_sampling_strategy = kwargs.get('timestep_sampling_strategy')
        
        # print("diffusion_noise_type: ", self.diffusion_noise_type)
        print("vision_denoising_type: ", self.vision_denoising_type)
        # print("timestep_sampling_strategy: ", self.timestep_sampling_strategy)
        # print("noise_scheduler_type: ", self.noise_scheduler_type)


    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """
        n_dim=clip_feats_x0.ndim
        dtype=clip_feats_x0.dtype
        """
        sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=timesteps.device, dtype=dtype)
        timesteps = timesteps.to(device=timesteps.device, dtype=dtype)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma  # torch.Size([2, 1, 1])


    def add_noise_or_mask(self, input_embs):
        if self.vision_denoising_type == 'continuous_diffusion':
            return self.add_diffusion_noise(input_embs)

        # if self.vision_denoising_type == 'discrete_diffusion':

        #     # create MLM mask and labels
        #     input_ids_img, labels_img, loss_weight, mask_prob = self.mask_or_random_replace_tokens(
        #         image_tokens,
        #         mask_id=self.mask_token_id,
        #         mask_schedule=get_mask_chedule("cosine"), # TODO: cosine schedule by default
        #         min_masking_rate=min_masking_rate,
        #         is_train=True,
        #     ) # input_ids_img: torch.Size([2, 256])   labels_img: torch.Size([2, 256])

        #     input_ids_t2i[:, -(self.num_vq_tokens + 1):-1] = input_ids_img # prepare input_ids
        #     input_embs_t2i = self.showo.model.embed_tokens(input_ids_t2i)

        #     labels_t2i[:, -(self.num_vq_tokens + 1):-1] = labels_img # prepare labels
        #     timesteps = None


    # def add_diffusion_noise(self, input_embs):
    #     if self.diffusion_noise_type == 'random_gaussian':
    #         # 1. timestep
    #         bsz = input_embs.shape[0]
    #         if self.timestep_sampling_strategy == 'logit_normal':
    #             u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,)
    #             indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
    #             timesteps = self.noise_scheduler.timesteps[indices].to(device=input_embs.device)
    #             # sigmas = self.get_sigmas(timesteps, n_dim=input_embs.ndim, dtype=input_embs.dtype)  # torch.Size([bsz, 1, 1, 1])
    #             sigmas = 0.001*timesteps[:, None, None, None]
    #         elif self.timestep_sampling_strategy == 'uniform':
    #             timesteps = torch.rand(bsz).to(input_embs.device) * 1000
    #             timesteps = torch.clamp(timesteps, min=1e-5) # to avoid numerical instability
    #             sigmas = 0.001*timesteps[:, None, None, None]
                
    #             timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=input_embs.device)
    #         # 2. noised inputs
    #         noise = torch.randn_like(input_embs, dtype=input_embs.dtype, device=input_embs.device) # torch.Size([2, 1568, 1024])
    #         noisy_input_embs = (sigmas * noise + (1.0 - sigmas) * input_embs)  #  torch.Size([2, 1568, 1024]) # no grad   # torch.Size([2, 2, 7, 512])
    #         return noisy_input_embs, noise, timesteps, sigmas


    def add_diffusion_noise(self, input_embs):
        bsz = input_embs.shape[0]
        noise = torch.randn_like(input_embs, dtype=input_embs.dtype, device=input_embs.device) # torch.Size([2, 1568, 1024])
        if self.noise_scheduler_type == "FlowMatchEulerDiscreteScheduler":
            u = compute_density_for_timestep_sampling(weighting_scheme=None, batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,)
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(device=input_embs.device)
            sigmas = self.get_sigmas(timesteps, n_dim=input_embs.ndim, dtype=input_embs.dtype)
            noisy_input_embs = (1.0 - sigmas) * input_embs + sigmas * noise
            # weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
        elif self.noise_scheduler_type == "DDPMScheduler":
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=input_embs.device)
            sigmas = 0.001*timesteps[:, None, None, None]
            noisy_input_embs = self.noise_scheduler.add_noise(input_embs, noise, timesteps).to(dtype=input_embs.dtype)
        return noisy_input_embs, noise, timesteps, sigmas


        # # if self.diffusion_noise_type == 'random_gaussian':
        # # 1. timestep
        # bsz = input_embs.shape[0]
        # if self.timestep_sampling_strategy == 'logit_normal':
        #     u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,)
        #     indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        #     timesteps = self.noise_scheduler.timesteps[indices].to(device=input_embs.device)
        #     # sigmas = self.get_sigmas(timesteps, n_dim=input_embs.ndim, dtype=input_embs.dtype)  # torch.Size([bsz, 1, 1, 1])
        #     sigmas = 0.001*timesteps[:, None, None, None]
        # elif self.timestep_sampling_strategy == 'uniform':
        #     # timesteps = torch.rand(bsz).to(input_embs.device) * 1000
        #     # timesteps = torch.clamp(timesteps, min=1e-5) # to avoid numerical instability
        #     timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=input_embs.device)
        #     sigmas = 0.001*timesteps[:, None, None, None]
        # # 2. noised inputs
        # noise = torch.randn_like(input_embs, dtype=input_embs.dtype, device=input_embs.device) # torch.Size([2, 1568, 1024])
        # if self.noise_scheduler_type == "FlowMatchEulerDiscreteScheduler":
        #     noisy_input_embs = (sigmas * noise + (1.0 - sigmas) * input_embs)  #  torch.Size([2, 1568, 1024]) # no grad   # torch.Size([2, 2, 7, 512])
        # elif self.noise_scheduler_type == "DDPMScheduler":
        #     noisy_input_embs = self.noise_scheduler.add_noise(input_embs, noise, timesteps).to(dtype=input_embs.dtype)
        # return noisy_input_embs, noise, timesteps, sigmas



        # elif self.diffusion_noise_type == 'mask_embedding':
        #     bsz = input_embs.shape[0]
        #     # 1. timestep
        #     if self.timestep_sampling_strategy == 'logit_normal':
        #         u = compute_density_for_timestep_sampling(weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,)
        #         indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        #         timesteps = self.noise_scheduler.timesteps[indices].to(device=input_embs.device)
        #         sigmas = self.get_sigmas(timesteps, n_dim=input_embs.ndim, dtype=input_embs.dtype)  # torch.Size([bsz, 1, 1])
        #     elif self.timestep_sampling_strategy == 'uniform':
        #         timesteps = torch.rand(bsz).to(input_embs.device) * 1000
        #         timesteps = torch.clamp(timesteps, min=1e-5) # to avoid numerical instability
        #         sigmas = 0.001*timesteps[:, None, None]
        #     elif self.timestep_sampling_strategy == 'full_mask':
        #         timesteps = torch.ones(bsz).to(input_embs.device) * 1000
        #         sigmas = 0.001*timesteps[:, None, None]
        #     # 2. noised inputs
        #     noise = self.showo.model.embed_tokens(torch.tensor(self.mask_token_id).to(input_embs.device))[None, None, :]
        #     noisy_input_embs = (sigmas * noise + (1.0 - sigmas) * input_embs)  #  torch.Size([2, 1568, 1024]) # no grad   # torch.Size([2, 2, 7, 512])
        #     return noisy_input_embs, noise, timesteps, sigmas

