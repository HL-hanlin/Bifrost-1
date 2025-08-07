import os
import io
import json
import math
from typing import Dict, Optional, Sequence, List, Union

import dataclasses
from dataclasses import dataclass, field

from google.cloud import storage

import logging
from ezcolorlog import root_logger as logger
logger.setLevel(logging.WARNING)

import torch
from torch import nn

IS_XLA_AVAILABLE = False
try:
    import torch_xla
    IS_XLA_AVAILABLE = True
except ImportError:
    pass

XLA_DISABLE_FUNCTIONALIZATION = bool(os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


@torch.no_grad()
def _shard_parameters_(self, params_to_shard) -> None:
    """
    At initialization we wrap a module with full parameters and shard the
    parameters in-place. Sharding is implemented by viewing each parameter
    as a 1D Tensor and retaining only a single slice, where the slice size
    is determined by the number of data parallel workers.

    Wrapping modules with many small parameters (or with a very large data
    parallel world size) will result in many small parameter shards and slow
    performance. In this case it's better to set *``flatten_parameters``* to
    ``True``, so that all of the small parameters in the module are combined
    into a single contiguous Tensor and sharded once.

    After this initial sharding is complete, the user can initialize a
    ``torch.optim.Optimizer`` in the usual way, i.e.::

    .. code-block:: python

        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    The optimizer will see only a single slice of parameters and will thus
    allocate less memory for optimizer state, avoiding redundancy across
    data parallel workers.

    Note: this method is implemented in a different manner from
    ``fairscale.nn.FullyShardedDataParallel``. Here we delete the original
    module parameters and create new sharded parameter tensors (instead of
    making sharded tensors an attribute of the original parameters). This
    make it easier to handle things (e.g. freeing parameters) on XLA.
    """

    #print_rank0("I actually use this to shard models!")
    if len(params_to_shard) > 0:
      # When freeing the full parameters, we point their internal XLATensor to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, dtype=self.compute_dtype, device=self.xla_device)

    # get the module names of each full parameter to shard
    params_to_shard_set = set(params_to_shard)
    assert len(params_to_shard_set) == len(params_to_shard), \
        "params_to_shard should not have dups"
    full_param_infos = []
    shared_full_param_memo = {}
    shared_full_param_infos = []
    full_params = []
    for module_name, m in self.named_modules():
      for n, p in m.named_parameters(recurse=False):
        if p.dtype != torch.float32:
          #raise TypeError("only fp32 parameters are supported")
          p.data = p.data.to(torch.float32)
        if p in params_to_shard_set:
          if p in shared_full_param_memo:
            mname, shared_m, shared_n = shared_full_param_memo[p]
            shared_full_param_infos.append(
                (module_name, mname, m, n, shared_m, shared_n))
          else:
            shared_full_param_memo[p] = (module_name, m, n)
            full_param_infos.append((module_name, m, n))
            full_params.append(p)
    assert len(full_params) == len(params_to_shard_set), \
        f"there are parameters in params_to_shard not belonging to this module."
    del shared_full_param_memo
    self.full_params = full_params
    self.full_param_infos = full_param_infos
    self.shared_full_param_infos = shared_full_param_infos

    # allocate and register new sharded parameters
    self.sharded_params = []
    for idx, (module_name, m, n) in enumerate(self.full_param_infos):
        p = self.full_params[idx]
        assert not hasattr(p, "_is_sharded")

        shard_data = self._get_shard(p)

        if shard_data.device != self.xla_device:
            # cast to XLA device if not already on XLA
            shard_data = shard_data.to(self.xla_device)
        p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
        p_shard._is_sharded = True
        p_shard._orig_size = p.size()
        p_shard._orig_name = f"{module_name}.{n}"
        p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
            ".", "_FSDP_SHARD_SEPARATOR_")
        self.register_parameter(p_shard._name, p_shard)
        self.sharded_params.append(p_shard)
        if p.device != self.xla_device:
            # cast to XLA device if not already on XLA
            p = p.to(self.xla_device).requires_grad_(p.requires_grad)
            # update p in full_params since id(p) changed after the casting
            self.full_params[idx] = p
        # Free the full parameter storage (here we free its internal XLATensor) but keep the tensor itself
        # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
        if XLA_DISABLE_FUNCTIONALIZATION:
            p.data = p.new_zeros(1)  # Old behavior before Functionalization.
        elif IS_XLA_AVAILABLE:
            import torch_xla
            torch_xla._XLAC._replace_xla_tensor(p, p.new_zeros(1))
        else:
            raise RuntimeError("XLA is not available")
        p._sharded_param = p_shard  # add a handle to the sharded parameter
        p._has_full_param = False
        # deregister the full parameter tensors from their modules (so that they won't
        # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
        # only the sharded parameters should appear in the FSDP model's `parameters()`)
        assert n in m._parameters
        m._parameters.pop(n)
        object.__setattr__(m, n, p)

    # also deregister the shared parameters
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
        assert n in m._parameters
        m._parameters.pop(n)
        shared_p = getattr(shared_m, shared_n)
        object.__setattr__(m, n, shared_p)

    assert len(self.sharded_params) == len(self.full_params)



import os
import wget
import torch
from tqdm import tqdm 

def _load_from_checkpoint(resume_from_checkpoint, local_checkpoint_path, model=None, num_ckpts=0):

    if num_ckpts == 0: # not use FSDP
        _rank = [0]
    else: # use FSDP
        _rank = list(range(num_ckpts))

    for r in _rank: # download ckpts from GCS
        SHARD_NAME = f'weights_rank-{r:08d}-of-{num_ckpts:08d}-pytorch_model.bin'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint, SHARD_NAME)
        local_checkpoint_name = os.path.join(local_checkpoint_path, SHARD_NAME)
        if os.path.exists(local_checkpoint_name):
            print(f"\n ===> {local_checkpoint_name} already exists...\n")
        else:
            os.makedirs(local_checkpoint_path, exist_ok=True)
            wget.download(SHARD_NAME_PATH, local_checkpoint_name)

    if num_ckpts == 0: # not use FSDP
        state_dict = torch.load(local_checkpoint_name)
        state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        return model
    else:

        state_dict_list = []
        for r in tqdm(_rank): # load model weights from checkpoints
            SHARD_NAME = f'weights_rank-{r:08d}-of-{num_ckpts:08d}-pytorch_model.bin'
            SHARD_NAME_PATH = os.path.join(resume_from_checkpoint, SHARD_NAME)
            local_checkpoint_name = os.path.join(local_checkpoint_path, SHARD_NAME)
            state_dict = torch.load(local_checkpoint_name)["model"]
            # remove FSDP prefixes and FSDP_SHARD_SEPARATOR
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                cleaned_key = key.replace("_FSDP_SHARD_SEPARATOR_", ".").replace("._fsdp_wrapped_module.", ".").replace("_fsdp_shard.", "").replace("_fpw_module.", "").replace("_fsdp_wrapped_module.", "")
                cleaned_state_dict[cleaned_key] = value
            state_dict_list.append(cleaned_state_dict)
        
        def get_weight(model, key):
            keys = key.split('.')  # Split the string into parts
            attr = model
            for k in keys:
                attr = getattr(attr, k)  # Traverse the model hierarchy
            return attr

        # now start combining shards into a single state dict
        all_keys = list(cleaned_state_dict.keys())
        from collections import OrderedDict
        model_weights = OrderedDict()
        for key in all_keys:
            # if "visual_encoder." not in key:
            target_weight = get_weight(model, key)
            # print(target_weight.shape)
            if target_weight.shape[0] == num_ckpts * cleaned_state_dict[key].shape[0]: # if dim 0 can be divided by num_ckpts
                collected_weights = [state_dict_list[r][key] for r in _rank]
                combined_weights = torch.cat(collected_weights, dim = 0)
                model_weights[key] = combined_weights
            elif target_weight.shape == cleaned_state_dict[key].shape: # if all shards keeps the same copy of this model weight
                # print(f"all shards should keep the same copy of the model weight with key {key}, {cleaned_state_dict[key].shape}")
                for i in _rank:
                    assert torch.all(cleaned_state_dict[key] == state_dict_list[r][key]).item(), f"ERROR: all shards should keep the same copy of the model weight with key {key}"
                model_weights[key] = cleaned_state_dict[key] # load with rank 0 since the weights are the same across all ranks
            else: # if dim 0 cannot be divided by num_ckpts, we truncate until the pos that's equal to the target size
                # print(key, "shape not match", target_weight.shape)
                collected_weights = [state_dict_list[r][key] for r in _rank]
                combined_weights = torch.cat(collected_weights, dim = 0)
                trunc_weights = combined_weights[:target_weight.shape[0]]
                assert torch.all(combined_weights[target_weight.shape[0]:] == 0).item(), "ERROR: the remaining positions should be all zero"
                model_weights[key] = trunc_weights
            # print(key, "finished")

        # sanity check 
        for key in model_weights.keys():
            # if (get_weight(model, key) == model_weights[key]).min() == False:
                # print(f"{key} in model: {get_weight(model, key)}")
                # print(f"{key} in state dict: {model_weights[key]}")
            # print(key, (get_weight(model, key) == model_weights[key]).min())
            assert get_weight(model, key).shape == model_weights[key].shape, f"ERROR: {key} has shape {get_weight(model, key).shape} in model, but {model_weights[key].shape} in state dict"

        model.load_state_dict(model_weights, strict=False)
        return model





from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
)
from transformers.configuration_utils import PretrainedConfig
from transformers import __version__
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model

def _load_from_checkpoint_mllm(resume_from_checkpoint, model, is_fsdp_enabled=True, save_safetensors=True, muted_keys=[]):

    """
    # resume_from_checkpoint = "/home/ubuntu/bansallab/hanlin/output_dir/multiflow_pub/Apr12C_tpu128_QwenVL3B_AddVisionBranch_t2i32_lr5e4_warm5k_step50k_MAR_LinearVisionHead_MSELoss_448res_256tokens_LearnPosEmb/checkpoint-38000"
    resume_from_checkpoint = '/home/ubuntu/bansallab/hanlin/output_dir/multiflow_pub/Apr30B_lambda1_QwenVL3B_AddVisionBranch_t2i128_lr5e4_warm5k_step50k_MAR_LinearVisionHead_CLIPLoss_224res_64tokens_LearnPosEmb/checkpoint-10000'
    is_fsdp_enabled = True
    save_safetensors = True
    """
    
    FSDP_MODEL_NAME = "pytorch_model_fsdp"

    config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
    adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
    adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
    weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
    weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
    safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
    safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
    is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
        # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
        any(
            FSDP_MODEL_NAME in folder_name
            for folder_name in os.listdir(resume_from_checkpoint)
            if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
        )
        # this checks the FSDP state dict when `FULL_STATE_DICT` is used
        or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
    )

    # if is_fsdp_ckpt and not is_fsdp_enabled:
    #     raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

    if not (
        any(
            os.path.isfile(f)
            for f in [
                weights_file,
                safe_weights_file,
                weights_index_file,
                safe_weights_index_file,
                adapter_weights_file,
                adapter_safe_weights_file,
            ]
        )
        or is_fsdp_ckpt
    ):
        raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

    logger.info(f"Loading model from {resume_from_checkpoint}.")

    if os.path.isfile(config_file):
        config = PretrainedConfig.from_json_file(config_file)
        checkpoint_version = config.transformers_version
        if checkpoint_version is not None and checkpoint_version != __version__:
            logger.warning(
                f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                f"Transformers but your current version is {__version__}. This is not recommended and could "
                "yield to errors or unwanted behaviors."
            )

    # if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
    #     # If the model is on the GPU, it still works!
    #     if is_sagemaker_mp_enabled():
    #         if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
    #             # If the 'user_content.pt' file exists, load with the new smp api.
    #             # Checkpoint must have been saved with the new smp api.
    #             smp.resume_from_checkpoint(
    #                 path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
    #             )
    #         else:
    #             # If the 'user_content.pt' file does NOT exist, load with the old smp api.
    #             # Checkpoint must have been saved with the old smp api.
    #             if hasattr(self.args, "fp16") and self.args.fp16 is True:
    #                 logger.warning(
    #                     "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
    #                 )
    #             state_dict = torch.load(
    #                 weights_file,
    #                 map_location="cpu",
    #                 weights_only=is_torch_greater_or_equal_than_1_13,
    #             )
    #             # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
    #             state_dict["_smp_is_partial"] = False
    #             load_result = model.load_state_dict(state_dict, strict=True)
    #             # release memory
    #             del state_dict
    #     elif self.is_fsdp_enabled:
    #         load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, resume_from_checkpoint)
    #     else:
    #         # We load the model state dict on the CPU to avoid an OOM error.
    #         if self.args.save_safetensors and os.path.isfile(safe_weights_file):
    #             state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
    #         else:
    #             state_dict = torch.load(
    #                 weights_file,
    #                 map_location="cpu",
    #                 weights_only=is_torch_greater_or_equal_than_1_13,
    #             )

    #         # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
    #         # which takes *args instead of **kwargs
    #         load_result = model.load_state_dict(state_dict, False)
    #         # release memory
    #         del state_dict
    #         self._issue_warnings_after_load(load_result)

    # # Load adapters following PR # 24096
    # elif _is_peft_model(model):
    #     # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
    #     if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
    #         if os.path.exists(resume_from_checkpoint):
    #             model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
    #         else:
    #             logger.warning(
    #                 "The intermediate checkpoints of PEFT may not be saved correctly, "
    #                 f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
    #                 "Check some examples here: https://github.com/huggingface/peft/issues/96"
    #             )
    #     else:
    #         logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
    # else:
    if True:
        # We load the sharded checkpoint
        load_result = load_sharded_checkpoint(model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=save_safetensors)
        if not is_sagemaker_mp_enabled():
            # self._issue_warnings_after_load(load_result)
            if len(load_result.missing_keys) != 0:
                if model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                    model._keys_to_ignore_on_save
                ):
                    model.tie_weights()
                else:
                    missing_keys_filter_out_visual = [key for key in load_result.missing_keys]
                    missing_keys_filter_out_visual = [key for key in missing_keys_filter_out_visual if "vision_language_model.visual." not in key]
                    for muted_key in muted_keys:
                        missing_keys_filter_out_visual = [key for key in missing_keys_filter_out_visual if muted_key not in key]
                        
                    logger.warning(f"There were missing keys in the checkpoint model loaded: {missing_keys_filter_out_visual}.")
            if len(load_result.unexpected_keys) != 0:
                logger.warning(
                    f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
                )





def count_params(params, text=None):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    if text is not None:
        print(text, total_trainable_params_count*1e-6, "M")
    else:
        print("total_trainable_params_count is: ", total_trainable_params_count*1e-6, "M")




# re-write TrainerState 
@dataclass
class MyTrainerState:
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    train_batch_size: int = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None
    stateful_callbacks: List["TrainerCallback"] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        # if self.stateful_callbacks is None:
        #     self.stateful_callbacks = {}
        # elif isinstance(self.stateful_callbacks, dict):
        #     # We are loading the callbacks in from the state file, no need to process them
        #     pass
        # else:
        #     # Saveable callbacks get stored as dict of kwargs
        #     stateful_callbacks = {}
        #     for callback in self.stateful_callbacks:
        #         if not isinstance(callback, (ExportableState)):
        #             raise TypeError(
        #                 f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
        #             )
        #         name = callback.__class__.__name__
        #         if name in stateful_callbacks:
        #             # We can have multiple versions of the same callback
        #             # if so, we store them as a list of states to restore
        #             if not isinstance(stateful_callbacks[name], list):
        #                 stateful_callbacks[name] = [stateful_callbacks[name]]
        #             stateful_callbacks[name].append(callback.state())
        #         else:
        #             stateful_callbacks[name] = callback.state()
        #     self.stateful_callbacks = stateful_callbacks

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                text = f.read()
            return cls(**json.loads(text))
            # logger.warning("tranerstate json_path is not in local machine, re-try to load from gcs")
        except:
            client = storage.Client()
            bucket = client.get_bucket('multiflow_pub')
            blob = bucket.blob(json_path)
            blob_bytes = blob.download_as_bytes()
            buffer = io.BytesIO(blob_bytes)
            trainer_state = cls(**json.load(buffer))
            # logger.warning("successfully loaded trainer state from gcs")
            return trainer_state
    
    # def compute_steps(self, args, max_steps):
    #     """
    #     Calculates and stores the absolute value for logging,
    #     eval, and save steps based on if it was a proportion
    #     or not.
    #     """
    #     for step_kind in ("logging", "eval", "save"):
    #         num_steps = getattr(args, f"{step_kind}_steps")
    #         if num_steps is not None:
    #             if num_steps < 1:
    #                 num_steps = math.ceil(max_steps * num_steps)
    #             setattr(self, f"{step_kind}_steps", num_steps)

    # def init_training_references(self, trainer, max_steps, num_train_epochs, trial):
    #     """
    #     Stores the initial training references needed in `self`
    #     """
    #     if trainer.hp_name is not None and trainer._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.trial_name = trainer.hp_name(trainer._trial)
    #     self.trial_params = None
    #     if trial is not None:
    #         from transformers.integrations import hp_params

    #         assignments = trial.assignments if trainer.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.trial_params = hp_params(assignments)

    #     self.max_steps = max_steps
    #     self.num_train_epochs = num_train_epochs
    #     self.is_local_process_zero = trainer.is_local_process_zero()
    #     self.is_world_process_zero = trainer.is_world_process_zero()



### we change "from collections import Mapping" into "from collections.abc import Mapping" to avoid ImportError: cannot import name 'Mapping' from 'collections'

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
import re
import six


def merge(left, right):
    """
    Merge two mappings objects together, combining overlapping Mappings,
    and favoring right-values

    left: The left Mapping object.
    right: The right (favored) Mapping object.

    NOTE: This is not commutative (merge(a,b) != merge(b,a)).
    """
    merged = {}

    left_keys = frozenset(left)
    right_keys = frozenset(right)

    # Items only in the left Mapping
    for key in left_keys - right_keys:
        merged[key] = left[key]

    # Items only in the right Mapping
    for key in right_keys - left_keys:
        merged[key] = right[key]

    # in both
    for key in left_keys & right_keys:
        left_value = left[key]
        right_value = right[key]

        if (isinstance(left_value, Mapping) and
                isinstance(right_value, Mapping)):  # recursive merge
            merged[key] = merge(left_value, right_value)
        else:  # overwrite with right value
            merged[key] = right_value

    return merged


@six.add_metaclass(ABCMeta)
class Attr(Mapping):
    """
    A mixin class for a mapping that allows for attribute-style access
    of values.

    A key may be used as an attribute if:
     * It is a string
     * It matches /^[A-Za-z][A-Za-z0-9_]*$/ (i.e., a public attribute)
     * The key doesn't overlap with any class attributes (for Attr,
        those would be 'get', 'items', 'keys', 'values', 'mro', and
        'register').

    If a values which is accessed as an attribute is a Sequence-type
    (and is not a string/bytes), it will be converted to a
    _sequence_type with any mappings within it converted to Attrs.

    NOTE: This means that if _sequence_type is not None, then a
        sequence accessed as an attribute will be a different object
        than if accessed as an attribute than if it is accessed as an
        item.
    """
    @abstractmethod
    def _configuration(self):
        """
        All required state for building a new instance with the same
        settings as the current object.
        """

    @classmethod
    def _constructor(cls, mapping, configuration):
        """
        A standardized constructor used internally by Attr.

        mapping: A mapping of key-value pairs. It is HIGHLY recommended
            that you use this as the internal key-value pair mapping, as
            that will allow nested assignment (e.g., attr.foo.bar = baz)
        configuration: The return value of Attr._configuration
        """
        raise NotImplementedError("You need to implement this")

    def __call__(self, key):
        """
        Dynamically access a key-value pair.

        key: A key associated with a value in the mapping.

        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __add__(self, other):
        """
        Add a mapping to this Attr, creating a new, merged Attr.

        other: A mapping.

        NOTE: Addition is not commutative. a + b != b + a.
        """
        if not isinstance(other, Mapping):
            return NotImplemented

        return self._constructor(merge(self, other), self._configuration())

    def __radd__(self, other):
        """
        Add this Attr to a mapping, creating a new, merged Attr.

        other: A mapping.

        NOTE: Addition is not commutative. a + b != b + a.
        """
        if not isinstance(other, Mapping):
            return NotImplemented

        return self._constructor(merge(other, self), self._configuration())

    def _build(self, obj):
        """
        Conditionally convert an object to allow for recursive mapping
        access.

        obj: An object that was a key-value pair in the mapping. If obj
            is a mapping, self._constructor(obj, self._configuration())
            will be called. If obj is a non-string/bytes sequence, and
            self._sequence_type is not None, the obj will be converted
            to type _sequence_type and build will be called on its
            elements.
        """
        if isinstance(obj, Mapping):
            obj = self._constructor(obj, self._configuration())
        elif (isinstance(obj, Sequence) and
              not isinstance(obj, (six.string_types, six.binary_type))):
            sequence_type = getattr(self, '_sequence_type', None)

            if sequence_type:
                obj = sequence_type(self._build(element) for element in obj)

        return obj

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.

        A key may be used as an attribute if:
         * It is a string
         * It matches /^[A-Za-z][A-Za-z0-9_]*$/ (i.e., a public attribute)
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
            isinstance(key, six.string_types) and
            re.match('^[A-Za-z][A-Za-z0-9_]*$', key) and
            not hasattr(cls, key)
        )


@six.add_metaclass(ABCMeta)
class MutableAttr(Attr, MutableMapping):
    """
    A mixin class for a mapping that allows for attribute-style access
    of values.
    """
    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping.
        """
        super(MutableAttr, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.

        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableAttr, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping.
        """
        super(MutableAttr, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.

        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableAttr, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )


class AttrDict(dict, MutableAttr):
    """
    A dict that implements MutableAttr.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

        self._setattr('_sequence_type', tuple)
        self._setattr('_allow_invalid_attributes', False)

    def _configuration(self):
        """
        The configuration for an attrmap instance.
        """
        return self._sequence_type

    def __getstate__(self):
        """
        Serialize the object.
        """
        return (
            self.copy(),
            self._sequence_type,
            self._allow_invalid_attributes
        )

    def __setstate__(self, state):
        """
        Deserialize the object.
        """
        mapping, sequence_type, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_sequence_type', sequence_type)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    def __repr__(self):
        return six.u('AttrDict({contents})').format(
            contents=super(AttrDict, self).__repr__()
        )

    @classmethod
    def _constructor(cls, mapping, configuration):
        """
        A standardized constructor.
        """
        attr = cls(mapping)
        attr._setattr('_sequence_type', configuration)

        return attr