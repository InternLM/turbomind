import logging
import os

from transformers import AutoConfig

from .base import BaseAWQForCausalLM
from .llama import LlamaAWQForCausalLM

AWQ_CAUSAL_LM_MODEL_MAP = {
    'llama': LlamaAWQForCausalLM,
}


def check_and_get_model_type(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir,
                                        trust_remote_code=True,
                                        **model_init_kwargs)
    if config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    return model_type


class AutoAWQForCausalLM:

    def __init__(self):
        raise EnvironmentError(
            'You must instantiate AutoAWQForCausalLM with\n'
            'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.'
            'from_pretrained')

    @classmethod
    def from_quantized(
        self,
        quant_path,
        max_seq_len=2048,
        fuse_layers=True,
        batch_size=1,
        device_map='balanced',
        max_memory=None,
        offload_folder=None,
        download_kwargs=None,
        **config_kwargs,
    ) -> BaseAWQForCausalLM:
        os.environ['AWQ_BATCH_SIZE'] = str(batch_size)
        model_type = check_and_get_model_type(quant_path)

        if config_kwargs.get('max_new_tokens') is not None:
            max_seq_len = config_kwargs['max_new_tokens']
            logging.warning(
                'max_new_tokens argument is deprecated... gracefully '
                'setting max_seq_len=max_new_tokens.')

        return AWQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path,
            model_type,
            max_seq_len,
            fuse_layers=fuse_layers,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            download_kwargs=download_kwargs,
            **config_kwargs,
        )
