import torch
import torch.nn as nn

import numpy as np
import inspect
import random
import copy
from typing import Any, Dict, List, Optional, Tuple, Union


from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)

from bifrost.models.flux_cnet.pipeline_flux_controlnet import FluxControlNetPipeline
from bifrost.models.flux_cnet.controlnet_flux import FluxControlNetModel


class FLUXControlNetDiffusionDecoder(nn.Module):

    def __init__(self, config, **kwargs):

        super().__init__()

        self.config = config

        # Load the tokenizers
        # load clip tokenizer
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.revision,
        )

        # load t5 tokenizer
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=config.revision,
        )

        # load clip text encoder
        self.text_encoder_one = CLIPTextModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
        )
        # load t5 text encoder
        self.text_encoder_two = T5EncoderModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=config.revision, variant=config.variant
        )


        # if not kwargs['remove_vae']:
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            revision=config.revision,
            variant=config.variant,
        )

        self.flux_transformer = FluxTransformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=config.revision,
            variant=config.variant,
        )

        if config.controlnet_model_name_or_path:
            self.flux_controlnet = FluxControlNetModel.from_pretrained(config.controlnet_model_name_or_path)
        else:
            # we can define the num_layers, num_single_layers,
            self.flux_controlnet = FluxControlNetModel.from_transformer(
                self.flux_transformer,
                attention_head_dim=self.flux_transformer.config["attention_head_dim"],
                num_attention_heads=self.flux_transformer.config["num_attention_heads"],
                num_layers=config.num_double_layers,
                num_single_layers=config.num_single_layers,
                # input_cond_dim=config.input_cond_dim
                # input_cond_dim=kwargs['input_cond_dim'],
            )

        # from bifrost.models.flux_cnet.controlnet_flux import NHWCUpsampleBlock
        # self.flux_controlnet.up_sampling_block = NHWCUpsampleBlock(kwargs['input_cond_dim'])


        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.vae.requires_grad_(False)
        self.flux_transformer.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.flux_controlnet.train()

        self.tokenizer_max_length = (self.tokenizer_one.model_max_length if hasattr(self, "tokenizer_one") and self.tokenizer_one is not None else 77)

        # use some pipeline function
        self.flux_controlnet_pipeline = FluxControlNetPipeline(
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder_one,
            tokenizer=self.tokenizer_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer_2=self.tokenizer_two,
            transformer=self.flux_transformer,
            controlnet=self.flux_controlnet,
        )



    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32, device=None):
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma



    # modified from pipeline_flux_controlnet.py in diffusers library
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        dtype: Optional[torch.dtype] = None,
        # lora_scale: Optional[float] = None,
    ):

        """
        prompt = prompt_batch 
        prompt_2 = prompt_batch

        num_images_per_prompt = 1
        prompt_embeds = None 
        pooled_prompt_embeds = None 
        max_sequence_length = 512
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        # if self.text_encoder is not None:
        #     if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder, lora_scale)

        # if self.text_encoder_2 is not None:
        #     if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder_2, lora_scale)

        # dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # device = device or self._execution_device
        # dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # if isinstance(self, TextualInversionLoaderMixin):
        #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer_two(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_two(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_two.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_two(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_two.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        # device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # if isinstance(self, TextualInversionLoaderMixin):
        #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer_one(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_one(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_one.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder_one(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_one.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds


    def compute_embeddings(self, prompt_batch, proportion_empty_prompts, weight_dtype, is_train=True, device=None):
        
        # prompt_batch = [e['text'] for e in batch]
        # prompt_batch = batch[args.caption_column]

        """
        prompt_batch = image_labels[args.num_images_per_batch:]
        proportion_empty_prompts = 0.0
        weight_dtype = torch.bfloat16 
        is_train = False
        device = model.device
        """
        
        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
        prompt_batch = captions
        # prompt_embeds, pooled_prompt_embeds, text_ids = self.flux_controlnet_pipeline.encode_prompt(prompt_batch, prompt_2=prompt_batch)
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(prompt_batch, prompt_2=prompt_batch, device=device, dtype=weight_dtype)
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
        text_ids = text_ids.to(dtype=weight_dtype)

        # text_ids [512,3] to [bs,512,3]
        text_ids = text_ids.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
        # return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "text_ids": text_ids}
        return prompt_embeds, pooled_prompt_embeds, text_ids
