import gc
from typing import Dict, Union

import torch
import torch.nn as nn
import transformers
from accelerate.big_modeling import (init_empty_weights,
                                     load_checkpoint_and_dispatch)
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from typing_extensions import Annotated, Doc

from ._config import AwqConfig
from .act import ScaledActivation
from .gemm import WQLinear_GEMM
from .module import (exclude_layers_to_not_quantize, get_named_linears,
                     set_op_by_name)

# from turbomind import Linear
# from turbomind.utils import turbomind_post_init

# Since we support different `AutoModelForxxx` from transformers
# we need to define a custom mapping dict as below:
TRANSFORMERS_AUTO_MAPPING_DICT = {
    'llama': 'AutoModelForCausalLM',
}


class BaseAWQForCausalLM(nn.Module):

    def __init__(
        self,
        model,
        model_type,
        is_quantized,
        config,
        quant_config,
    ):
        """The base model for all AutoAWQ models.

        Args:
            model: The pretrained or quantized model.
            model_type: The model type, found in config.json.
            is_quantized: Indicates if the current model is quantized
            config: The config of the model.
            quant_config: The quantization config of the model.
        """
        super().__init__()
        self.model: PreTrainedModel = model
        self.model_type: str = model_type
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config: AwqConfig = quant_config

    def to(self, device: Annotated[str,
                                   Doc('The device to move your model to.')]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    # @staticmethod
    # def fuse_layers(model):
    #     pass

    @classmethod
    def from_quantized(self,
                       model_path: str,
                       model_type: str,
                       max_seq_len: int,
                       torch_dtype: torch.dtype = torch.float16,
                       device_map: Union[str, Dict] = 'balanced',
                       **config_kwargs: Dict):
        """A method for initialization of a quantized model, usually in INT4.

        Args:
            model_path (str): The model path
            model_type (str): The model type, loaded from config.json.
            max_seq_len (int): The maximum sequence cached sequence length of
                the model. Larger values may increase loading time and
                memory usage.
            torch_dtype: The dtype to load the model as. May not work with
                other values than float16.
            device_map: A device map that will be passed onto the model
                loading method from transformers.
        **config_kwargs: Additional kwargs that are passed to the config
            during initialization
        """
        # [STEP 1-2] Load weights path and configs
        model_weights_path, config, quant_config = self._load_config(
            self,
            model_path,
            max_seq_len=max_seq_len,
            **config_kwargs,
        )

        target_cls_name = TRANSFORMERS_AUTO_MAPPING_DICT[config.model_type]
        target_cls = getattr(transformers, target_cls_name)

        # [STEP 3] Load model
        with init_empty_weights():
            model = target_cls.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        # Prepare WQLinear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
            quant_config.version,
            use_exllama=False,
            use_exllama_v2=False,
            use_qbits=False,
        )

        model.tie_weights()

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_weights_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            dtype=torch_dtype,
        )

        # TODO
        # model = turbomind_post_init(model)

        # # Dispatch to devices
        # if fuse_layers:
        #     self.fuse_layers(model)

        model.eval()

        return self(
            model,
            model_type,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
        )

    def _load_config(
        self,
        model_path,
        max_seq_len=4096,
        **config_kwargs,
    ):
        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config = AwqConfig.from_pretrained(model_path)

        # Load model config and set max generation length
        if max_seq_len is None and hasattr(self, 'max_seq_len_key'):
            config = AutoConfig.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                **config_kwargs)
            config.max_seq_len = getattr(config, self.max_seq_len_key, 2048)
            # To add the generate support for Multi-modal models as well
            if hasattr(config, 'text_config'):
                config.text_config.max_seq_len = getattr(
                    config, self.max_seq_len_key, 2048)
        else:
            max_seq_len = 2048 if max_seq_len is None else max_seq_len
            config = AutoConfig.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                **config_kwargs)
            config.max_seq_len = max_seq_len

        return model_path, config, quant_config

    def _load_quantized_modules(self,
                                model,
                                quant_config,
                                version,
                                use_exllama,
                                use_exllama_v2,
                                use_qbits=False):
        # Real quantization of weights
        assert not (version == 'gemv' and
                    (use_exllama or use_exllama_v2 or
                     use_qbits)), 'Exllama kernels only support GEMM version.'

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc='Replacing layers...'):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert)

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                assert version == 'gemm'

                q_linear_module = WQLinear_GEMM
                # q_linear_module = Linear
                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size,
                    True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            if not use_qbits:
                torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict['is_scalable']:
            if not isinstance(scale_dict['scale_layer'], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(scale_dict['scale_shape'],
                                        dtype=param.dtype,
                                        device=param.device)

                # scale activation
                scaled_act = ScaledActivation(scale_dict['scale_layer'],
                                              scale_like)
                set_op_by_name(layer, scale_dict['scale_name'], scaled_act)
