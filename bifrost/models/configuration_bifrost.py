# Copyright (c) 2023-2024 Han Lin.
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

from bifrost.utils import AttrDict

from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig

from transformers.utils import logging
logger = logging.get_logger(__name__)


class VisionGenerationEncoderConfig(PretrainedConfig):
    model_type = "vision_gen_enc"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationDecoderConfig(PretrainedConfig):
    model_type = "vision_gen_dec"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationVAEConfig(PretrainedConfig):
    model_type = "vision_gen_vae"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationTokenizerConfig(PretrainedConfig):
    model_type = "vision_gen_tokenizer"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationAlignerConfig(PretrainedConfig):
    model_type = "vision_gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationHeadConfig(PretrainedConfig):
    model_type = "vision_gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VLMConfig(PretrainedConfig):
    model_type = "vision_language_model"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class DiffusionDecoderConfig(PretrainedConfig):
    model_type = "diffusion_decoder"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.use_cache = False

        ##### vlm #####
        vision_language_model_config = kwargs.get("vision_language_model_config", {})
        self.vision_language_model_config = VLMConfig(**vision_language_model_config)

        ##### vae #####
        vision_gen_enc_config = kwargs.get("vision_gen_enc_config", {})
        self.vision_gen_enc_config = VisionGenerationEncoderConfig(**vision_gen_enc_config)
        vision_gen_dec_config = kwargs.get("vision_gen_dec_config", {})
        self.vision_gen_dec_config = VisionGenerationDecoderConfig(**vision_gen_dec_config)
        vision_gen_vae_config = kwargs.get("vision_gen_vae_config", {})
        self.vision_gen_vae_config = VisionGenerationVAEConfig(**vision_gen_vae_config)

        diffusion_decoder_config = kwargs.get("diffusion_decoder_config", {})
        self.diffusion_decoder_config = DiffusionDecoderConfig(**diffusion_decoder_config)

        ##### tokenizer #####
        vision_gen_tokenizer_config = kwargs.get("vision_gen_tokenizer_config")
        # self.vision_gen_tokenizer_config = VisionGenerationTokenizerConfig(**vision_gen_tokenizer_config)
        vision_gen_aligner_config = kwargs.get("vision_gen_aligner_config")
        # self.vision_gen_aligner_config = VisionGenerationAlignerConfig(**vision_gen_aligner_config)
        vision_gen_head_config = kwargs.get("vision_gen_head_config")
        # self.vision_gen_head_config = VisionGenerationHeadConfig(**vision_gen_head_config)

        ##### others #####
        # self.noise_scheduler_type = kwargs.get("noise_scheduler_type", None)
        # self.diffusion_noise_type = kwargs.get("diffusion_noise_type", None)
        # self.timestep_sampling_strategy = kwargs.get("timestep_sampling_strategy", None)
        self.vision_denoising_type = kwargs.get("vision_denoising_type", None)
        self.max_seq_length = kwargs.get("max_seq_length", None)
        self.num_visual_gen_tokens = kwargs.get("num_visual_gen_tokens", None)
        self.precise_prompt_mask = kwargs.get("precise_prompt_mask", 3)
        self.add_vision_branch = kwargs.get("add_vision_branch", False)
        self.add_vision_branch_reuse_layernorm = kwargs.get("add_vision_branch_reuse_layernorm", False)
        self.use_discrete_visual_tokenizer = kwargs.get("use_discrete_visual_tokenizer", False)
        self.add_vision_gen_mask_token = kwargs.get("add_vision_gen_mask_token", False)
        # self.dit_style_vision_branch = kwargs.get("dit_style_vision_branch", False)
        self.mar_style_vision_branch = kwargs.get("mar_style_vision_branch", False)
        
        
