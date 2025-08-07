import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import random 
from einops import rearrange

import torch 

from transformers import AutoProcessor

from bifrost.train.utils import prepare_4d_causal_attention_mask


@dataclasses.dataclass
class Conversation:

    def __init__(self, **kwargs):

        self.processor_name_or_path = kwargs.get("processor_name_or_path")
        self.processor = AutoProcessor.from_pretrained(self.processor_name_or_path)
        
        try:
            self.pad_id = self.processor.tokenizer.pad_token_id
        except:
            self.pad_id = self.processor.pad_token_id
        
        if self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
            self.image_pad = 151655 # <|image_pad|>
        elif self.processor_name_or_path in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct',
                                            'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-3B-Instruct', 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-14B-Instruct']:
            self.image_pad = 151655 # 151643 # use the same text pad
        
        self.ignore_id = kwargs.get("ignore_id", -100) # used as masks in loss calculation
        self.max_seq_length = kwargs.get("max_seq_length", 60) # 60 is only enough for imagenet, need to enlarge when training on open-world images with long prompts
        self.full_vision_mask = kwargs.get("full_vision_mask", False)
        self.precise_prompt_mask = kwargs.get("precise_prompt_mask", False)
        self.add_timestep_token = kwargs.get("add_timestep_token", True)
        self.cond_dropout_prob = kwargs.get("cond_dropout_prob", 0.0)
        self.add_vision_soi_eoi_tokens = kwargs.get("add_vision_soi_eoi_tokens", False)
        self.add_vision_soi_token = kwargs.get("add_vision_soi_token", False)
        # self.use_separate_2d_rope_for_vision = kwargs.get("use_separate_2d_rope_for_vision", False)
        # self.use_1d_rope_for_vision = kwargs.get("use_1d_rope_for_vision", False)
        self.vision_pos_emb_type = kwargs.get("vision_pos_emb_type", '2drope')


    # adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = 1 # TODO: hard coded: WE USE 1 HERE INSTEAD OF 2 in QWEN2.5-VL!    self.config.vision_config.spatial_merge_size # 2
        image_token_id = 151655 # TODO: hard coded. self.config.image_token_id # 151655
        video_token_id = 151656 # TODO: hard coded. self.config.video_token_id # 151656
        vision_start_token_id = 151652 # TODO: hard coded.self.config.vision_start_token_id # 151652
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids # torch.Size([1, 3602])
            if attention_mask is None: # torch.Size([1, 3602])
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device) # torch.Size([1, 3602])
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1] # torch.Size([3602])
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1) # tensor([14], device='cuda:0')
                vision_tokens = input_ids[vision_start_indices + 1] # tensor([151655], device='cuda:0')
                image_nums = (vision_tokens == image_token_id).sum() # tensor(1, device='cuda:0')
                video_nums = (vision_tokens == video_token_id).sum() # tensor(0, device='cuda:0')
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st) # 15
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1 # 3603
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        ) # 1, 98, 146
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1 # 0
                        ed = ed_image # 15

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    ) # 1, 49, 73
                    text_len = ed - st # 15

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0 # 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    """
                    llm_pos_ids_list = [tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])]
                    """
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1) # tensor([[0]])
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w) # tensor([[0, 0, 0,  ..., 0, 0, 0]])      torch.Size([1, 3577])

                    time_tensor = expanded_range * second_per_grid_t * 2 # TODO: hard coded. self.config.vision_config.tokens_per_second # torch.Size([1, 3577])

                    time_tensor_long = time_tensor.long() # tensor([[0, 0, 0,  ..., 0, 0, 0]])      torch.Size([1, 3577])
                    t_index = time_tensor_long.flatten() # tensor([0, 0, 0,  ..., 0, 0, 0])         torch.Size([3577])

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten() # tensor([ 0,  0,  0,  ..., 48, 48, 48])     torch.Size([3577])
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten() # tensor([ 0,  1,  2,  ..., 70, 71, 72])     torch.Size([3577])
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    """
                    llm_pos_ids_list = [tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]]),
                                        tensor([[15, 15, 15,  ..., 15, 15, 15],
                                                [15, 15, 15,  ..., 63, 63, 63],
                                                [15, 16, 17,  ..., 85, 86, 87]])]
                    """
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w # 3592

                if st < len(input_tokens): # True
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0 # tensor(88)
                    text_len = len(input_tokens) - st # 10
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    """
                    llm_pos_ids_list = [tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                                                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]]),
                                        tensor([[15, 15, 15,  ..., 15, 15, 15],
                                                [15, 15, 15,  ..., 63, 63, 63],
                                                [15, 16, 17,  ..., 85, 86, 87]]),
                                        tensor([[88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
                                                [88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
                                                [88, 89, 90, 91, 92, 93, 94, 95, 96, 97]])]
                    """

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                """
                llm_positions = tensor([[ 0,  1,  2,  ..., 95, 96, 97],
                                        [ 0,  1,  2,  ..., 95, 96, 97],
                                        [ 0,  1,  2,  ..., 95, 96, 97]])
                """
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device) # torch.Size([3, 1, 3602])
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def remove_suffix(self, prompt):
        suffix_to_remove = [
            'The image shows', 
            'In the image,', 
            'The image depicts', 
            'The image captures', 
            'The image features', 
            'The image is',
            'The image displays',
            'The image presents',
            'The image showcases',
            "The image you've provided appears to be",
            "This is an image of",
            "The image appears to be",
            "This image features",
            "This image depicts",
            "This image is",
            "This is",
            "This aerial image showcases",
            "The image portrays",
            "The image you've provided is",
        ]
        clean_prompt = []
        for p in prompt:
            for suffix in suffix_to_remove:
                if p.startswith(suffix):
                    p = p[len(suffix):]
                    p = p.strip()
                    p = p.capitalize()
                    break
            clean_prompt.append(p)
        return clean_prompt



    def t2i_prompt(self, 
        prompt, 
        img_h=256,
        img_w=256, 
        num_visual_gen_tokens=256,
        ): # TODO: need to support different resolutions later

        """
        prompt = ['miniature schnauzer', 'American chameleon']
        num_visual_gen_tokens = 256
        img_h=256
        img_w=256
        i=0

        self = uni_prompting

        """
        
        if type(prompt) == str:
            prompt = [prompt]

        n_prompts = len(prompt)
        texts = []

        prompt = self.remove_suffix(prompt)

        for i in range(n_prompts):

            if random.random() < self.cond_dropout_prob:
                prompt_after_cond_drop = ""
            else:
                prompt_after_cond_drop = prompt[i]

            if self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]: 
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                # "text": f"Generate an image of height {img_h} and width {img_w} that adheres to the following description. {prompt[i]}"
                                "text": f"Generate an image of {prompt_after_cond_drop}"
                                # "text": f"{prompt[i]}"
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            # {
                            #     "type": "text", 
                            #     "text": "Here is an image based on your request:"
                            # },
                            {
                                "type": "image",
                                "image": None, # this will create a <image_pad> token, which will be used as placeholder for timestep emb later
                            },
                        ],
                    }
                ]
                suffix_length = 0 if self.precise_prompt_mask else 3
                subtract_time_token_length = 0  # if self.add_timestep_token else 1
            elif self.processor_name_or_path in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct',
                                            'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-3B-Instruct', 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-14B-Instruct']:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": f"Generate an image that adheres to the following description. {prompt_after_cond_drop}"},
                ]
                suffix_length = 0
                subtract_time_token_length = 0 

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
            # trim_substring = "<|im_end|>\n<|im_start|>assistant\n"
            # if text.endswith(trim_substring):
            #     text = text[:-len(trim_substring)]  # Trim the last part of the string
            if self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
                if self.precise_prompt_mask:
                    # trim_substring = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                    # if text.startswith(trim_substring):
                    #     text = text[len(trim_substring):]
                    # text = text.replace("<|im_end|>\n<|im_start|>assistant\n", ". ")
                    # text = text.replace("<|im_end|>\n", "")
                    # text = text.replace("Generate an image of ", "")
                    # text = text.replace("Here is an image based on your request:", "")
                    # text = text.replace("<|image_pad|><|vision_end|>", "")
                    # text = "An image of " + text

                    # trim_substring = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    # if text.startswith(trim_substring):
                    #     text = text[len(trim_substring):]
                    
                    trim_substring = "<|image_pad|><|vision_end|><|im_end|>\n"
                    if text.endswith(trim_substring):
                        text = text[:-len(trim_substring)]
                # if self.add_vision_soi_token:
                #     trim_substring = "<|vision_start|>"
                #     if text.endswith(trim_substring):
                #         text = text[:-len(trim_substring)]
                    
            elif self.processor_name_or_path in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct',
                                'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-3B-Instruct', 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-14B-Instruct']:
                if self.precise_prompt_mask:
                    # print("here")
                    text = text.replace("system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nGenerate an image that adheres to the following description. ", "")
                    trim_substring = "<|vision_start|>"
                    if not text.endswith(trim_substring):
                        text = text + trim_substring
            texts.append(text)



        # inputs = self.processor(text=texts, padding='max_length', return_tensors="pt", max_length=self.max_seq_length, padding_side = 'left')
        inputs = self.processor(text=texts, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_seq_length, padding_side = 'left')
        
        inputs_postfix = self.processor(text="<|im_end|>\n<|im_start|>assistant\n<|vision_start|>", truncation=True, return_tensors="pt", max_length=self.max_seq_length, padding_side = 'left')
        input_ids_postfix = inputs_postfix['input_ids']
        input_ids = inputs['input_ids']
        input_ids[:,-input_ids_postfix.shape[1]:] = input_ids_postfix

        attention_mask = inputs['attention_mask']

        vision_soi_eoi_token_length = 2 if self.add_vision_soi_eoi_tokens else 0
        vision_soi_token_length = 1 if self.add_vision_soi_token else 0



        if self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
            input_ids = torch.cat([
                input_ids, 
                torch.tensor([self.image_pad] * (num_visual_gen_tokens - subtract_time_token_length + vision_soi_eoi_token_length + vision_soi_token_length)).unsqueeze(0).repeat(n_prompts, 1), # note that the original <image_pad> token in message is used as placeholder for timestep emb. So here we concat 151655 * num_visual_gen_tokens for visual tokens
                # input_ids[:,-suffix_length:], # the last THREE tokens are <|vision_end|> <|im_end|> \n
                ], dim=1)

            attention_mask = torch.cat([
                attention_mask, 
                torch.tensor([1] * (num_visual_gen_tokens - subtract_time_token_length + vision_soi_eoi_token_length + vision_soi_token_length)).unsqueeze(0).repeat(n_prompts, 1),
                # attention_mask[:,-suffix_length:], 
                ], dim=1)
        else:
            input_ids = torch.cat([
                input_ids, 
                torch.tensor([self.image_pad] * (num_visual_gen_tokens - subtract_time_token_length + vision_soi_eoi_token_length + vision_soi_token_length)).unsqueeze(0).repeat(n_prompts, 1), # note that the original <image_pad> token in message is used as placeholder for timestep emb. So here we concat 151655 * num_visual_gen_tokens for visual tokens
                ], dim=1)

            attention_mask = torch.cat([
                attention_mask, 
                torch.tensor([1] * (num_visual_gen_tokens - subtract_time_token_length + vision_soi_eoi_token_length + vision_soi_token_length)).unsqueeze(0).repeat(n_prompts, 1),
                ], dim=1)

        # get rope position_ids 
        patch_size = int(num_visual_gen_tokens**0.5)
        image_grid_thw = torch.tensor((1, patch_size, patch_size), dtype=torch.int64, device=input_ids.device).repeat(n_prompts, 1)
        
        # print("input_ids conversation", input_ids.shape)

        position_ids, _ = self.get_rope_index(
                input_ids, 
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=attention_mask,
            ) 

        # print("position_ids conversation", position_ids.shape)

        if self.vision_pos_emb_type == '1drope':
            # if self.use_1d_rope_for_vision: #  and self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct"]:
            position_ids[:,:,-num_visual_gen_tokens:] = position_ids[0,:,:].max(axis = -1).values.unsqueeze(1) + torch.arange(num_visual_gen_tokens)
        elif self.vision_pos_emb_type == 'separate_2drope': #self.use_separate_2d_rope_for_vision:
            # if self.processor_name_or_path not in ["Qwen/Qwen2.5-VL-3B-Instruct"]:
            #     position_ids[:1,:,-num_visual_gen_tokens:] = position_ids[0,:,:60].max(axis = -1).values.unsqueeze(1)  + 1
            #     position_ids[1:,:,-num_visual_gen_tokens:] = position_ids[0,:,:60].max(axis = -1).values.unsqueeze(1) + torch.arange(num_visual_gen_tokens) + 1
            # else:
            position_ids[1:, :, -num_visual_gen_tokens:] -= position_ids[0,:,-num_visual_gen_tokens:].unsqueeze(0)
            position_ids[0,:,-num_visual_gen_tokens:] = 0
        elif self.vision_pos_emb_type == '2drope': # nothing needed if so
            pass 
        elif self.vision_pos_emb_type == 'learnable_pos_emb': # no rotation for vision
            position_ids[:, :, -num_visual_gen_tokens:] = 0 

    


        attention_mask = prepare_4d_causal_attention_mask(attention_mask)

        if self.full_vision_mask: ### TODO remember to add this in inference code
            if suffix_length > 0:
                attention_mask[:,:,-(num_visual_gen_tokens+suffix_length):-suffix_length, -(num_visual_gen_tokens+suffix_length):-suffix_length] = 0.0
            else:
                attention_mask[:,:,-num_visual_gen_tokens:, -num_visual_gen_tokens:] = 0.0

        image_position_mask = torch.zeros((n_prompts, attention_mask.shape[-1]), dtype=torch.int64, device=attention_mask.device) # TODO: currently regarding time & <start/end> as text embeddings
        if self.processor_name_or_path in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
            if suffix_length > 0:
                image_position_mask[:, -(num_visual_gen_tokens+suffix_length):-suffix_length] = 1
            else:
                image_position_mask[:, -(num_visual_gen_tokens+vision_soi_token_length):] = 1    
        elif self.processor_name_or_path in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct',
                                                    'Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-3B-Instruct', 'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-14B-Instruct']:
            image_position_mask[:, -(num_visual_gen_tokens + vision_soi_eoi_token_length + vision_soi_token_length):] = 1                      
        

        if self.add_vision_soi_token and self.processor_name_or_path not in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]:
            input_ids[:,self.max_seq_length] = 151652 # "<|vision_start|>"

        labels = input_ids
        labels = torch.where(labels == self.pad_id, self.ignore_id, labels)
        labels = torch.where(labels == self.image_pad, self.ignore_id, labels)
        # labels = torch.where(labels == self.pad_id, self.ignore_id, labels)


        # print("position_ids conversation start", position_ids.shape)
        position_ids = rearrange(position_ids, "k bsz c -> bsz k c")
        # print("position_ids conversation end", position_ids.shape)

        return input_ids, labels, attention_mask, image_position_mask, position_ids


    
