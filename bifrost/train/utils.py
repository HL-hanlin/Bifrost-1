import math
import random
import numpy as np 
from PIL import Image
from typing import Any, List, Tuple, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import torch.nn.functional as F
from torchvision import transforms



##################################################
#              training utils
##################################################

def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True):
    batch_size, seq_len = image_tokens.shape

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob






def image_transform(image, processor_type, resolution=None, resize_short_side_to_resolution=False, resize_long_side_to_resolution=False, random_flip=False):

    """
    resolution=256
    image = Image.open("/Users/hanlin/Documents/GitHub/multiflow_qwen/debug/debug_output.png").convert("RGB")
    resize_short_side_to
    """

    if processor_type == 'stabilityai/sdxl-vae' or processor_type == 'black-forest-labs/FLUX.1-dev' or processor_type == 'pretrained_models/vae/kl16.ckpt' or processor_type == 'vit_so400m_patch14_siglip_384':

        if resize_short_side_to_resolution: # Resize short side to resolution, maintaining aspect ratio
            w, h = image.size
            scale = resolution / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            image = transforms.CenterCrop((resolution, resolution))(image)

        if processor_type == 'vit_so400m_patch14_siglip_384':
            image = image.resize((384, 384), Image.BICUBIC)

        if random_flip and random.random() < 0.5:
            image = transforms.RandomHorizontalFlip(p=1.0)(image)

        image = torch.from_numpy((np.array(image) / 127.5) - 1) # torch.Size([384, 256, 3]), [-1, 1]
        image = image.permute(2, 0, 1) # torch.Size([3, 384, 256]), c h w

    return image



    # if mode == "showo-default":
    #     image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    #     image = transforms.CenterCrop((resolution, resolution))(image)

    # elif mode == "multimodal":
    #     # Resize long side to 384, maintaining aspect ratio
    #     w, h = image.size
    #     scale = resolution / max(w, h)
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     image = image.resize((new_w, new_h), Image.BICUBIC)
    #     # Pad the shorter side to 384
    #     pad_w, pad_h = (resolution - new_w) // 2, (resolution - new_h) // 2
    #     padding = (pad_w, pad_h, resolution - new_w - pad_w, resolution - new_h - pad_h)
    #     image = transforms.functional.pad(image, padding, fill=(127, 127, 127), padding_mode="constant")

    # elif mode == "generation":
    #     # Resize short side to 384, maintaining aspect ratio
    #     w, h = image.size
    #     scale = resolution / min(w, h)
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     image = image.resize((new_w, new_h), Image.BICUBIC)
    #     # Center-crop the long side to 384
    #     image = transforms.CenterCrop((resolution, resolution))(image)

    # # Convert to tensor
    # image = transforms.ToTensor()(image)

    # # Normalize if needed
    # if normalize:
    #     image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)

    # return image


def prepare_4d_causal_attention_mask(attention_mask):
    batch_size, sequence_length = attention_mask.shape
    # min_dtype = torch.finfo(torch.float32).min
    min_dtype = torch.finfo(torch.bfloat16).min
    # min_dtype = torch.finfo(attention_mask.dtype).min
    causal_mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=torch.float32)
    diagonal_attend_mask = torch.arange(sequence_length) > torch.arange(sequence_length).reshape(-1, 1)
    causal_mask *= diagonal_attend_mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        if attention_mask.shape[-1] > sequence_length:
            attention_mask = attention_mask[:, :sequence_length]
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    return causal_mask
