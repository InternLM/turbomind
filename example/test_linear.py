import torch
import torch.nn as nn
from safetensors import safe_open

import turbomind as tm
from turbomind.utils import unpack_awq_gemm

torch.manual_seed(0)


def dequantize(qweight, qzeros, scales, group_size: int = 128):
    _qweight = unpack_awq_gemm(qweight)
    _qzeros = unpack_awq_gemm(qzeros)
    _qzeros = _qzeros.half()
    weight = _qweight.clone().half()
    for i in range(qzeros.shape[0]):
        start = i * group_size
        end = start + group_size
        weight[start:end] = (weight[start:end, :] -
                             _qzeros[i:i + 1, :]) * scales[i:i + 1, :]
    return weight


def load_specified_linear_weights():
    ckpt_path = '/models/140/llama3/Meta-Llama-3-8B-Instruct-hf-AWQ/model-00001-of-00002.safetensors'  # noqa
    layer_id = 0
    # prefix = f'model.layers.{layer_id}.self_attn.q_proj.'
    prefix = f'model.layers.{layer_id}.self_attn.o_proj.'
    keys = ['qweight', 'qzeros', 'scales']
    tensors = {}
    with safe_open(ckpt_path, framework='pt', device='cuda') as f:
        for key in keys:
            tensors[key] = f.get_tensor(prefix + key)

    return tensors


tensors = load_specified_linear_weights()
qweight, qzeros, scales = tensors['qweight'], tensors['qzeros'], tensors[
    'scales']

group_size = 128
in_features = qweight.shape[0]
out_features = qweight.shape[1] * 8

x = torch.randn(in_features, device=qweight.device, dtype=torch.float16)

weight = dequantize(qweight, qzeros, scales, group_size)
print(f'-- dequantization: weight.shape={weight.shape}, weight: \n{weight}')
ref_linear = nn.Linear(in_features, out_features, bias=False, device='cuda')
with torch.no_grad():
    ref_linear.weight = nn.Parameter(weight.T)
    ref_res = ref_linear(x)
    print(f'nn.linear.res: {ref_res}')

model = tm.Linear(in_features=in_features,
                  out_features=out_features,
                  bias=False,
                  quant_method='awq',
                  w_bit=4,
                  group_size=group_size)

model.qweight = qweight
model.qzeros = qzeros
model.scales = scales

model.post_init()

res = model(x)
print(f'tm.linear.res: {res}')
max_diff = torch.max(abs(ref_res - res))
ave_diff = torch.sum(abs(ref_res - res)) / ref_res.numel()
print(f'max_diff {max_diff}, ave_diff {ave_diff}')
