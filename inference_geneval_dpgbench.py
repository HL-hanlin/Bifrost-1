# coding=utf-8
# Copyright 2025 Han Lin (hanlincs@cs.unc.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

import shlex
import ast
import json
import io 
import math
import copy
import wandb
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from einops import rearrange
from google.cloud import storage

import dataclasses
from dataclasses import dataclass, field, fields

import typing
from typing import Dict, Optional, Sequence, List, Union, get_args

import transformers
from transformers import AutoTokenizer, AutoProcessor

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms

from bifrost.models.configuration_bifrost import MultiModalityConfig
from bifrost.models.modeling_bifrost import MultiModalityCausalLM
from bifrost.conversation import Conversation
from bifrost.models.diffusion import retrieve_timesteps
from bifrost.utils import _load_from_checkpoint, _load_from_checkpoint_mllm
from bifrost.train.utils import image_transform
from bifrost.models.mar.utils import sample_orders, random_masking

from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor


from torchvision.utils import make_grid
import torchvision.transforms as T



@dataclass
class TrainingArguments(transformers.TrainingArguments):

    ## model 
    vision_language_model: str = field(default='Qwen2_5_VLForConditionalGeneration')
    vision_language_model_name: str = field(default='Qwen/Qwen2.5-VL-7B-Instruct')
    vision_gen_vae: str = field(default=None)

    ## processing
    t2i_resolution: int = field(default=448) # 384
    num_visual_gen_tokens: int = field(default=256) 
    max_seq_length: int = field(default=300)

    ## inference 
    batch_size: int = field(default = 1) # inference batch size
    bf16: bool = field(default = True)
    cfg_weight: float = field(default=1.0) 
    cfg_schedule: str = field(default='linear') 
    temperature: float = field(default=1.0) 

    ## log, save, eval (HF trainer)
    huggingface_token: str = field(default=None)

    # t2i gen params
    timestep_sampling_strategy: str = field(default='uniform')
    vision_denoising_type: str = field(default='mar')
    add_timestep_token: bool = field(default=False)
    add_vision_gen_mask_token: bool = field(default=False)
    add_vision_soi_token: bool = field(default=False)
    add_vision_soi_eoi_tokens: bool = field(default=False)

    vision_head_type: str = field(default='linear')
    vision_loss_type: str = field(default='mse')
    vision_pos_emb_type: str = field(default='learnable_pos_emb')

    # masks
    full_vision_mask: bool = field(default=True)
    precise_prompt_mask: bool = field(default=True)
    add_vision_branch: bool = field(default=True)
    add_vision_branch_reuse_layernorm: bool = field(default=False)
    use_discrete_visual_tokenizer: bool = field(default=False)
    skip_text_part2: bool = field(default=True)
    proportion_empty_prompts: float = field(default=0.0)
    use_clip_visual_encoder: bool = field(default=True)
    lambda_gpu: bool = field(default = True)
    use_2d_query_tokens: bool = field(default=False)

    # e2d training 
    e2e_training: bool = field(default=False) 
    ctrlnet_training: bool = field(default=False) 
    pretrained_diffusion_decoder_name_or_path: str = field(default="black-forest-labs/FLUX.1-dev")
    num_single_layers: int = field(default=1)
    num_double_layers: int = field(default=4)
    diffusion_decoder_text_dropout_prob: float = field(default=0.0)
    vae_w_ctrlnet_training: bool = field(default=False)
    vae_wo_ctrlnet_training: bool = field(default=False)
    vae_scale_by_4: bool = field(default=False)

    eval_with_prompt: bool = field(default=True)
    eval_dpgbench: bool = field(default=True)
    eval_geneval: bool = field(default=True)
    num_images_per_batch: int = field(default=4)

    output_dir: str = field(default='./output')
    local_checkpoint_path: str = field(default = None)
    
    

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking



@dataclass
class DataCollatorForGenEval(object):

    def __init__(self, uni_prompting, args = None):

        self.args = args
        self.uni_prompting = uni_prompting

        self.t2i_image_processor = None 
        if args.vision_language_model == 'Qwen2_5_VLForConditionalGeneration':
            self.t2i_image_processor = AutoProcessor.from_pretrained(args.vision_language_model_name)

        file_path = "evaluation/GenEval/prompts/generation_prompts.txt"
        # file_path = "evaluation/GenEval/prompts/geneval_rewrite_generation_prompts.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.split("\n")
        text = [t for t in text if len(t)>0]
        self.prompts = text

        file_path = "evaluation/GenEval/prompts/evaluation_metadata.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            self.json = [json.loads(line) for line in f if line.strip()]


    def prepare_t2i(self, texts):

        input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i = self.uni_prompting.t2i_prompt(
            texts, 
            img_h=self.args.t2i_resolution,
            img_w=self.args.t2i_resolution,
            num_visual_gen_tokens=self.args.num_visual_gen_tokens
        )
        
        return input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i


    def __call__(self, i) -> Dict[str, torch.Tensor]:

        image_grid_thw = torch.tensor([self.args.num_images_per_batch, self.args.t2i_resolution//14, self.args.t2i_resolution//14]) #.repeat(2,1)

        t2i_input_ids = [""] * self.args.num_images_per_batch + [self.prompts[i]] * self.args.num_images_per_batch 
        input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i = self.prepare_t2i(t2i_input_ids)
        
        batch = {}
        batch['t2i_flow'] = {
            "input_ids": input_ids_t2i,
            "labels": labels_t2i,
            "attention_mask": attention_mask_t2i,
            'image_position_mask': image_position_mask_t2i,
            "position_ids": position_ids_t2i,
            "t2i_image_grid_thw": image_grid_thw,
            'ar_mask': torch.ones((2*self.args.num_images_per_batch, self.args.num_visual_gen_tokens)),
            'image_labels': t2i_input_ids,
            }

        return batch



@dataclass
class DataCollatorForDPGBench(object):

    def __init__(self, uni_prompting, args = None):

        self.args = args
        self.uni_prompting = uni_prompting

        self.t2i_image_processor = None 
        if args.vision_language_model == 'Qwen2_5_VLForConditionalGeneration':
            self.t2i_image_processor = AutoProcessor.from_pretrained(args.vision_language_model_name)

        self.prompts_folder = sorted(os.listdir("evaluation/DPGBench/dpg_bench/prompts"))
        self.prompts = []

        for i in range(len(self.prompts_folder)):
            with open(os.path.join("evaluation/DPGBench/dpg_bench/prompts", self.prompts_folder[i]), 'r', encoding='utf-8') as f:
                prompt = f.read()
            self.prompts.append(prompt)


    def prepare_t2i(self, texts):

        input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i = self.uni_prompting.t2i_prompt(
            texts, 
            img_h=self.args.t2i_resolution,
            img_w=self.args.t2i_resolution,
            num_visual_gen_tokens=self.args.num_visual_gen_tokens
        )
        
        return input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i


    def __call__(self, i) -> Dict[str, torch.Tensor]:

        prompt = self.prompts[i]

        image_grid_thw = torch.tensor([self.args.num_images_per_batch, self.args.t2i_resolution//14, self.args.t2i_resolution//14]) #.repeat(2,1)

        t2i_input_ids = [""] * self.args.num_images_per_batch + [prompt] * self.args.num_images_per_batch 
        input_ids_t2i, labels_t2i, attention_mask_t2i, image_position_mask_t2i, position_ids_t2i = self.prepare_t2i(t2i_input_ids)
        
        batch = {}
        batch['t2i_flow'] = {
            "input_ids": input_ids_t2i,
            "labels": labels_t2i,
            "attention_mask": attention_mask_t2i,
            'image_position_mask': image_position_mask_t2i,
            "position_ids": position_ids_t2i,
            "t2i_image_grid_thw": image_grid_thw,
            'ar_mask': torch.ones((2*self.args.num_images_per_batch, self.args.num_visual_gen_tokens)),
            'image_labels': t2i_input_ids,
            }

        return batch



def save_image_sets(image_sets, idx, metadata, output_root):

    os.makedirs(output_root, exist_ok=True)
    image_list = image_sets

    # for idx, image_list in enumerate(image_sets):
    set_dir = os.path.join(output_root, f"{idx:05d}")
    samples_dir = os.path.join(set_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Save individual images
    for i, img in enumerate(image_list):
        img.save(os.path.join(samples_dir, f"{i:04d}.png"))

    # Save image grid (2x2)
    tensor_images = [T.ToTensor()(img) for img in image_list]
    grid = make_grid(tensor_images, nrow=2, padding=2)
    grid_img = T.ToPILImage()(grid)
    grid_img.save(os.path.join(set_dir, "grid.png"))

    # Save dummy metadata (you can modify this based on your actual needs)
    metadata_path = os.path.join(set_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        f.write(json.dumps(metadata) + "\n")



def save_image_sets_dpgbench(image_sets, img_text, output_root):

    os.makedirs(output_root, exist_ok=True)
    image_list = image_sets
    tensor_images = [T.ToTensor()(img) for img in image_list]
    grid = make_grid(tensor_images, nrow=2, padding=2)
    grid_img = T.ToPILImage()(grid)
    grid_img.save(os.path.join(output_root, img_text.replace(".txt", ".png")))



if __name__ == '__main__':

    parser = transformers.HfArgumentParser((TrainingArguments))
    args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ##### VLM #####
    vision_language_model_config = {
        "cls": args.vision_language_model,
        "model_type": "vision_language_model",
        "params": {
            'model_name_or_path': args.vision_language_model_name, # 3.08B for LLM, 668M for CLIP vision_und_enc
            "load_from_pretrained": True,
            'remove_vision_und_encoder': False,
            'frozen_modules_in_vlm': ["vision_language_model.lm_head", "vision_language_model.model.embed_tokens", "vision_language_model.model.layers", "vision_language_model.model.norm"],
            'huggingface_token': args.huggingface_token,
        },
    }


    vision_gen_vae_config = {
        "cls": "AutoencoderKL", "model_type": "vision_gen_vae",
        "params": {
            'model_name_or_path': 'black-forest-labs/FLUX.1-dev', # 83M params for FLUX VAE
            "load_from_pretrained": True,
            'frozen_vision_gen_vae': False,
            'huggingface_token': args.huggingface_token,
        },
    }


    if args.vision_language_model_name in ['Qwen/Qwen2.5-VL-3B-Instruct', "Qwen/Qwen2.5-3B-Instruct"]:
        n_embed = 2048
    elif args.vision_language_model_name in ['Qwen/Qwen2.5-VL-7B-Instruct', "Qwen/Qwen2.5-7B-Instruct"]:
        n_embed = 3584


    ##### Diffusion Decoder #####
    diffusion_decoder_config = {
        "cls": args.pretrained_diffusion_decoder_name_or_path,
        "model_type": "diffusion_decoder",
        "params": {
            'pretrained_model_name_or_path': args.pretrained_diffusion_decoder_name_or_path, 
            "revision": None,
            "variant": None,
            "controlnet_model_name_or_path": None,
            'num_single_layers': args.num_single_layers,
            'num_double_layers': args.num_double_layers,
            "diffusion_decoder_text_dropout_prob": args.diffusion_decoder_text_dropout_prob,
        },
    }


    model_config = {
        'vision_language_model_config': vision_language_model_config,
        'vision_gen_vae_config': vision_gen_vae_config,
        "diffusion_decoder_config": diffusion_decoder_config,
        'timestep_sampling_strategy': args.timestep_sampling_strategy,
        'vision_denoising_type': args.vision_denoising_type,
        'max_seq_length': args.max_seq_length,
        'num_visual_gen_tokens': args.num_visual_gen_tokens,
        "add_vision_branch": args.add_vision_branch,
        "add_vision_branch_reuse_layernorm": args.add_vision_branch_reuse_layernorm,
        "use_discrete_visual_tokenizer": args.use_discrete_visual_tokenizer,
        "add_timestep_token": args.add_timestep_token,
        "skip_text_part2": args.skip_text_part2,
        "add_vision_gen_mask_token": args.add_vision_gen_mask_token,
        "add_vision_soi_eoi_tokens": args.add_vision_soi_eoi_tokens,
        "add_vision_soi_token": args.add_vision_soi_token,
        "vision_head_type": args.vision_head_type,
        "vision_loss_type": args.vision_loss_type,
        "vision_pos_emb_type": args.vision_pos_emb_type,
        "use_clip_visual_encoder": args.use_clip_visual_encoder,
        "batch_size_t2i": 1,
        "t2i_resolution": args.t2i_resolution,
        "lambda_gpu": args.lambda_gpu,
        "use_2d_query_tokens": args.use_2d_query_tokens,
        "e2e_training": True,
        "ctrlnet_training": args.ctrlnet_training,
        "proportion_empty_prompts": 0.0,
        "lambda_clip": 0.0,
        "remove_vae": False,
        "vae_w_ctrlnet_training": args.vae_w_ctrlnet_training,
        "vae_wo_ctrlnet_training": args.vae_wo_ctrlnet_training,
        "inner_dim": n_embed,
        "vae_scale_by_4": args.vae_scale_by_4
    }

    model_config = MultiModalityConfig(**model_config)
    model = MultiModalityCausalLM(model_config)

    _load_from_checkpoint_mllm(args.local_checkpoint_path, model, muted_keys = ['diffusion_decoder'])

    model.eval()
    model = model.to(torch.bfloat16).to(device)
    # autocast_ctx = torch.autocast(model.device.type)


    ###########################
    #  model -- FLUX CtrlNet  #
    ###########################

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", revision=None, variant=None)
    flux_transformer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", revision=None, variant=None)

    from bifrost.models.flux_cnet.pipeline_flux_controlnet import FluxControlNetPipeline

    # if args.use_diffusers_ctrlnet:
    from bifrost.models.flux_cnet.controlnet_flux import FluxControlNetModel
    flux_controlnet = FluxControlNetModel.from_transformer(
        flux_transformer,
        attention_head_dim=flux_transformer.config["attention_head_dim"],
        num_attention_heads=flux_transformer.config["num_attention_heads"],
        num_layers=1,
        num_single_layers=4,
    )

    flux_controlnet = flux_controlnet.from_pretrained(args.local_checkpoint_path, subfolder='flux_controlnet')

    flux_controlnet_pipeline = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=flux_controlnet,
        transformer=flux_transformer,
        torch_dtype=torch.float32,
    )

    flux_controlnet_pipeline.enable_model_cpu_offload()
    flux_controlnet_pipeline = flux_controlnet_pipeline.to(torch.bfloat16)

    # from diffusers.training_utils import free_memory
    # free_memory()



    #################
    #      data     #
    #################

    # set up processor
    conversation_config = {
        'processor_name_or_path': vision_language_model_config['params']['model_name_or_path'],
        'full_vision_mask': args.full_vision_mask,
        'precise_prompt_mask': args.precise_prompt_mask,
        "add_timestep_token": args.add_timestep_token,
        "cond_dropout_prob": 0.0,
        "add_vision_soi_eoi_tokens": args.add_vision_soi_eoi_tokens,
        "add_vision_soi_token": args.add_vision_soi_token,
        "vision_pos_emb_type": args.vision_pos_emb_type,
        "max_seq_length": args.max_seq_length
    }
    uni_prompting = Conversation(**conversation_config) ## TODO: need to set max_seq_length>60 for open-world image generation

    data_collator_kwargs = {"uni_prompting": uni_prompting, "args": args}

    if args.eval_geneval:
        data_collator = DataCollatorForGenEval(**data_collator_kwargs)
    elif args.eval_dpgbench:
        data_collator = DataCollatorForDPGBench(**data_collator_kwargs)



    #################
    #   inference   #
    #################

    ouput_dir = args.output_dir
    output_dir_w_prompt = os.path.join(ouput_dir, 'val_images_w_prompt')
    os.makedirs(output_dir_w_prompt, exist_ok=True)


    for i in tqdm(range(len(data_collator.prompts))):
    
        ##########################
        #####  part 1: MLLM  #####
        ##########################

        # if model.device.type == 'cpu':
        #     model = model.to('cuda')

        t2i_flow = data_collator(i)['t2i_flow']
        input_ids_t2i, labels, attention_mask, image_position_mask, position_ids, image_grid_thw, ar_mask, image_labels = t2i_flow["input_ids"], t2i_flow["labels"], t2i_flow["attention_mask"], t2i_flow['image_position_mask'], t2i_flow['position_ids'], t2i_flow['t2i_image_grid_thw'], t2i_flow['ar_mask'], t2i_flow['image_labels']                
        
        if position_ids.shape[1] == 3:
            position_ids = rearrange(position_ids, "bsz k c -> k bsz c")

        dtype, device = model.dtype, model.device
        bsz = args.num_images_per_batch

        with torch.no_grad():
            text_embs_part1 = model.vision_language_model.model.embed_tokens(input_ids_t2i[:, :-args.num_visual_gen_tokens].to(device)) 
            
        # initialize 
        mask = torch.ones(bsz, args.num_visual_gen_tokens).to(device) 
        tokens = model.mask_token.repeat(bsz, args.num_visual_gen_tokens, 1) 
        orders = sample_orders(bsz, args.num_visual_gen_tokens).to(device) 

        num_iter = args.num_visual_gen_tokens
        for step in tqdm(list(range(num_iter))):
            cur_tokens = tokens.clone() 
            tokens = torch.cat([tokens, tokens], dim=0) 
            mask = torch.cat([mask, mask], dim=0) 

            if args.vision_pos_emb_type == 'learnable_pos_emb':
                tokens = tokens + model.learnable_pos_emb
            
            input_embeddings = torch.cat((text_embs_part1, tokens), dim=1) 

            with torch.no_grad():
                outputs = model.vision_language_model(inputs_embeds=input_embeddings, 
                                        use_cache=None, 
                                        attention_mask=attention_mask.to(device),
                                        past_key_values=None,
                                        return_dict=True,
                                        position_ids = position_ids.to(device),
                                        image_position_mask=image_position_mask.to(device),
                                        t_emb = None
                                        )

            hidden_states = outputs['hidden_states'] 
            denoised_hidden_states = hidden_states[:, -args.num_visual_gen_tokens:] 

            if args.vision_head_type == 'linear':
                with torch.no_grad():
                    model_pred = model.vision_head(denoised_hidden_states) 

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(args.num_visual_gen_tokens * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(), torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, args.num_visual_gen_tokens)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            model_pred = model_pred[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if args.cfg_schedule == "linear":
                cfg_iter = 1 + (args.cfg_weight - 1) * (args.num_visual_gen_tokens - mask_len[0]) / args.num_visual_gen_tokens
            elif args.cfg_schedule == "constant":
                cfg_iter = args.cfg_weight

            sampled_token_latent = model_pred[:args.num_images_per_batch] + args.cfg_weight * (model_pred[args.num_images_per_batch:] - model_pred[:args.num_images_per_batch]) 
            mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # if model.device.type == 'cuda':
        #     model = model.cpu()

        # free_memory()

        ###########################
        ##### part 2: ctrlnet #####
        ###########################

        with torch.no_grad():
            prompt_embeds_cond, pooled_prompt_embeds_cond, text_ids_cond = flux_controlnet_pipeline.encode_prompt(image_labels[args.num_images_per_batch:], prompt_2=image_labels[args.num_images_per_batch:])

        ## ours 
        generator = torch.Generator(device=model.device).manual_seed(args.seed)
        with torch.no_grad():
            # with autocast_ctx:
            images_w_prompt = flux_controlnet_pipeline(
                prompt_embeds=prompt_embeds_cond,
                pooled_prompt_embeds=pooled_prompt_embeds_cond,
                control_image = tokens.detach(),
                num_inference_steps=28,
                controlnet_conditioning_scale=0.7,
                guidance_scale=3.5,
                generator=generator,
                height=args.t2i_resolution * 16 // 14, 
                width=args.t2i_resolution * 16 // 14,
            ).images


        for j in range(int(i*args.num_images_per_batch), int((i+1)*args.num_images_per_batch)):
            if args.eval_geneval:
                img_text = image_labels[args.num_images_per_batch + j-i*args.num_images_per_batch].replace(" ", "_")
                images_w_prompt[j-i*args.num_images_per_batch].save(os.path.join(output_dir_w_prompt, f"{j:06d}_{img_text}.png"))
            elif args.eval_dpgbench:
                img_text = data_collator.prompts_folder[i]
                images_w_prompt[j-i*args.num_images_per_batch].save(os.path.join(output_dir_w_prompt, img_text.replace(".txt", ".png")))
                images_w_prompt[j-i*args.num_images_per_batch].save(os.path.join(output_dir_w_prompt, f"{j:06d}_{img_text}.png"))


        if args.eval_geneval:
            metadata = data_collator.json[i]
            save_image_sets(images_w_prompt, i, metadata, os.path.join(os.path.dirname(output_dir_w_prompt), "geneval_bifrost"))
        elif args.eval_dpgbench:
            save_image_sets_dpgbench(images_w_prompt, img_text, os.path.join(os.path.dirname(output_dir_w_prompt), "dpgbench_bifrost"))
