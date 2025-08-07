# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import requests
from einops import rearrange
from tqdm import tqdm 
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from bifrost.models.diffusion import Diffusion

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory



def model_name_to_cls(cls_name):

    if "ShallowUViTEncoder" in cls_name:
        from bifrost.models.uvit import ShallowUViTEncoder
        cls = ShallowUViTEncoder
    elif "ShallowUViTDecoder" in cls_name:
        from bifrost.models.uvit import ShallowUViTDecoder
        cls = ShallowUViTDecoder
    elif 'AutoencoderKL' in cls_name:
        from bifrost.models.autoencoder.autoencoder_kl import AutoencoderKL
        cls = AutoencoderKL    
    elif 'Qwen2_5_VLForConditionalGeneration' in cls_name:
        from bifrost.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration 
        cls = Qwen2_5_VLForConditionalGeneration
    elif 'Qwen2ForCausalLM' in cls_name:
        from bifrost.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM 
        cls = Qwen2ForCausalLM
    elif 'magvitv2' in cls_name:
        from bifrost.models.magvitv2.modeling_magvitv2 import MAGVITv2
        cls = MAGVITv2
    elif "MlpProjector" in cls_name:
        from bifrost.models.projector import MlpProjector
        cls = MlpProjector
    elif "VisionHead" in cls_name:
        from bifrost.models.projector import VisionHead
        cls = VisionHead
    elif 'black-forest-labs/FLUX.1-dev' in cls_name:
        from bifrost.models.diffusion_decoder import FLUXControlNetDiffusionDecoder
        cls = FLUXControlNetDiffusionDecoder
    elif cls_name == None:
        return None
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls



class MultiModalityCausalLM(PreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.use_clip_visual_encoder = config.use_clip_visual_encoder
        self.vision_denoising_type = config.vision_denoising_type
        self.vision_head_type = config.vision_head_type
        self.vision_loss_type = config.vision_loss_type
        self.batch_size_t2i = config.batch_size_t2i 
        self.t2i_resolution = config.t2i_resolution
        self.num_visual_gen_tokens = config.num_visual_gen_tokens
        self.vision_pos_emb_type = config.vision_pos_emb_type
        self.lambda_gpu = config.lambda_gpu
        self.use_2d_query_tokens = config.use_2d_query_tokens
        self.e2e_training = config.e2e_training
        self.ctrlnet_training = config.ctrlnet_training
        self.remove_vae = config.remove_vae
        self.proportion_empty_prompts = config.proportion_empty_prompts
        self.lambda_clip = config.lambda_clip
        self.vae_w_ctrlnet_training = config.vae_w_ctrlnet_training
        self.vae_wo_ctrlnet_training = config.vae_wo_ctrlnet_training
        self.inner_dim = config.inner_dim
        self.config = config

        # vision-language model
        vision_language_model_config = config.vision_language_model_config
        vision_language_model_cls = model_name_to_cls(vision_language_model_config.cls)
        self.vision_language_model = vision_language_model_cls.from_pretrained(
            vision_language_model_config.params.model_name_or_path,
            torch_dtype= torch.bfloat16 # torch.bfloat16
            )

        if config.add_vision_branch:
            num_hidden_layers = self.vision_language_model.model.config.num_hidden_layers
            for i in range(num_hidden_layers):
                self.vision_language_model.model.layers[i].self_attn.add_vision_attention()
                self.vision_language_model.model.layers[i].add_vision_mlp()
                if not config.add_vision_branch_reuse_layernorm:
                    self.vision_language_model.model.layers[i].add_vision_layernorms()
            if not config.add_vision_branch_reuse_layernorm:
                self.vision_language_model.model.add_vision_norm()


        # diffusion module
        self.diffusion = Diffusion(config)

        # # vision generation vae
        vision_gen_vae_config = config.vision_gen_vae_config
        vision_gen_vae_model_cls = model_name_to_cls(vision_gen_vae_config.cls)
        if vision_gen_vae_config.params.model_name_or_path == 'black-forest-labs/FLUX.1-dev':
            self.vision_gen_vae_model = vision_gen_vae_model_cls.from_pretrained(
                vision_gen_vae_config.params.model_name_or_path, 
                subfolder='vae', 
                token=vision_gen_vae_config.params.huggingface_token
            )
            self.in_channels = 64

        if self.vision_denoising_type == 'mar': 
            self.in_channels = self.inner_dim
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.inner_dim))
            if self.vision_pos_emb_type == 'learnable_pos_emb':
                self.learnable_pos_emb = nn.Parameter(torch.zeros(1, self.num_visual_gen_tokens, self.inner_dim))    
            else:
                self.learnable_pos_emb = None

            # vision Loss
            if self.vision_head_type == 'linear': 
                self.vision_head = nn.Linear(self.inner_dim, self.in_channels, bias=True)


    # this forward function is only used for training
    def forward(
            self,
            lm_flow=None,
            t2i_flow=None,
            mmu_flow=None,
            **kwargs,
    ):

        max_seq_length = kwargs.get('max_seq_length')
        num_visual_gen_tokens = kwargs.get('num_visual_gen_tokens') * 4 if ((self.vae_wo_ctrlnet_training or self.vae_w_ctrlnet_training) and not self.config.vae_scale_by_4) else kwargs.get('num_visual_gen_tokens')
        label_smoothing = kwargs.get('label_smoothing')
        min_masking_rate = kwargs.get('min_masking_rate')
        log_task_specific_loss = kwargs.get('log_task_specific_loss')

        t2i_coeff = kwargs.get('t2i_coeff')
        lm_coeff = kwargs.get('lm_coeff')
        mmu_coeff = kwargs.get('mmu_coeff')
        
        batch_size_t2i = kwargs.get('batch_size_t2i', None)
        batch_size_lm = kwargs.get('batch_size_lm', None)
        batch_size_mmu = kwargs.get('batch_size_mmu', None)

        precise_prompt_mask = kwargs.get('precise_prompt_mask')
        add_vision_branch = kwargs.get("add_vision_branch")
        add_vision_branch_reuse_layernorm = kwargs.get("add_vision_branch_reuse_layernorm")
        add_timestep_token = kwargs.get("add_timestep_token")
        use_discrete_visual_tokenizer = kwargs.get("use_discrete_visual_tokenizer")
        skip_text_part2 = kwargs.get("skip_text_part2")
        add_vision_gen_mask_token = kwargs.get("add_vision_gen_mask_token")
        suffix_length = 1 if precise_prompt_mask else 3
        add_vision_soi_eoi_tokens = kwargs.get("add_vision_soi_eoi_tokens")
        add_vision_soi_token = kwargs.get("add_vision_soi_token")
        vision_soi_eoi_tokens_length = 1 if add_vision_soi_eoi_tokens else 0
        vision_soi_token_length = 1 if add_vision_soi_token else 0
        suffix_length = 0

        # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
        # Build formatted sequences for class-conditional/text-to-image generation
        # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*

        pixel_values_or_image_ids = t2i_flow["pixel_values"]
        image_clip_embs = t2i_flow['image_clip_embs'] if 'image_clip_embs' in t2i_flow else None
        input_ids_t2i = t2i_flow["input_ids"]
        labels = t2i_flow["labels"]
        attention_mask = t2i_flow["attention_mask"]
        image_position_mask = t2i_flow['image_position_mask']
        position_ids = t2i_flow['position_ids']
        image_grid_thw = t2i_flow['t2i_image_grid_thw']
        ar_mask = t2i_flow['ar_mask']
        t_emb = None
        device = pixel_values_or_image_ids.device
        position_ids = rearrange(position_ids, "bsz k c -> k bsz c")


        with torch.no_grad():
            if not hasattr(self, 'vision_gen_vae_model'): # lora FT clip encoder
                if self.lambda_gpu:
                    if image_clip_embs is None:
                        input_embs_img = self.vision_language_model.visual(pixel_values_or_image_ids.to(self.dtype), grid_thw=image_grid_thw, same_grid_images=False)
                        input_embs_img = rearrange(input_embs_img, "(b hw) c -> b hw c", b=pixel_values_or_image_ids.shape[0]) 
                    else:
                        input_embs_img = image_clip_embs.to(self.dtype)
                    if self.e2e_training or self.vae_w_ctrlnet_training or self.vae_wo_ctrlnet_training:
                        pixel_latents = self.diffusion_decoder.vae.encode(pixel_values_or_image_ids.to(self.dtype)).latent_dist.sample() # encode image with vae 
                else:
                    input_embs_img = self.vision_language_model.visual(pixel_values_or_image_ids.to(self.dtype), grid_thw=image_grid_thw, same_grid_images=False)
                    input_embs_img = rearrange(input_embs_img, "(b hw) c -> b hw c", b=pixel_values_or_image_ids.shape[0]) 

            elif self.vision_gen_vae_model.config._name_or_path == 'black-forest-labs/FLUX.1-dev':
                input_embs_img = self.vision_gen_vae_model.encode(pixel_values_or_image_ids.to(self.dtype)).latent_dist.sample() 
                input_embs_img = (input_embs_img - self.vision_gen_vae_model.config.shift_factor) * self.vision_gen_vae_model.config.scaling_factor
                vae_scale_factor = 2 ** (len(self.vision_gen_vae_model.config.block_out_channels) - 1)

         
        if self.vision_denoising_type in ['ar', 'mar', 'xar', 'flowar']: 
            z_emb = self.mask_token * ar_mask.unsqueeze(-1) + input_embs_img * (1-ar_mask.unsqueeze(-1)) 
            if self.vision_pos_emb_type == 'learnable_pos_emb':
                z_emb = z_emb + self.learnable_pos_emb


        time_token_length = 1 if add_timestep_token else 0
        if add_vision_soi_eoi_tokens:
            text_embs_part1 = self.vision_language_model.model.embed_tokens(input_ids_t2i[:, :-(num_visual_gen_tokens + suffix_length + time_token_length + 2 )]) 
        else:
            text_embs_part1 = self.vision_language_model.model.embed_tokens(input_ids_t2i[:, :-(num_visual_gen_tokens + suffix_length + time_token_length )]) 
        
        if skip_text_part2:
            if add_timestep_token:
                input_embeddings = torch.cat((text_embs_part1, t_emb.unsqueeze(1), z_emb), dim=1) 
            else:
                input_embeddings = torch.cat((text_embs_part1, z_emb), dim=1)
            if suffix_length > 0:
                attention_mask = attention_mask[:, :, :-suffix_length, :-suffix_length]
                position_ids = position_ids[:, :, :-suffix_length]
                image_position_mask = image_position_mask[:, :-suffix_length] if add_vision_branch is not None else None
    

        outputs = self.vision_language_model(inputs_embeds=input_embeddings.to(self.dtype), 
                                    use_cache=None, 
                                    attention_mask=attention_mask.to(self.dtype),
                                    past_key_values=None,
                                    return_dict=True,
                                    position_ids = position_ids,
                                    image_position_mask=image_position_mask if add_vision_branch is not None else None,
                                    t_emb = t_emb
                                    )
        hidden_states = outputs['hidden_states'] 
        

        if batch_size_t2i > 0 and not self.ctrlnet_training:
            if self.diffusion.vision_denoising_type == 'mar':
                if skip_text_part2:
                    denoised_hidden_states = hidden_states[:batch_size_t2i, -(num_visual_gen_tokens):]
                else:
                    denoised_hidden_states = hidden_states[:batch_size_t2i, -(num_visual_gen_tokens + suffix_length):-suffix_length] 

                bsz, seq_len, _ = input_embs_img.shape

                if self.vision_head_type == 'linear':
                    model_pred = self.vision_head(denoised_hidden_states)
                    target = input_embs_img

                if self.vision_loss_type == 'mse':
                    loss_t2i = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
               
        outputs = {'loss': loss_t2i}
        return outputs 
