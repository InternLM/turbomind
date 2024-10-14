import torch
import torch.nn as nn
from safetensors import safe_open

import turbomind as tm

torch.manual_seed(0)


def i32x8_to_i4x8(w):
    """merge 8 integers (range from 0 to 15) into one 32-bit integer."""
    assert w.shape[-1] % 8 == 0
    shape = (w.shape[0], w.numel() // (w.shape[0] * 8), 8)
    shape = shape[:-1] + (1, )
    result = torch.zeros(shape, dtype=w.dtype, device=w.device)
    mask = torch.tensor([15], dtype=w.dtype, device=w.device)
    for i in range(8):
        shift = 4 * (7 - i)
        result[..., 0] |= (w[..., i] & mask) << shift
    result = result.view(w.shape[0], -1)
    return result


def i4x8_to_i32x8(w):
    """split one integer every 4bits into 8 integers (range from 0 to 15)"""
    shape = w.shape + (8, )
    result = torch.zeros(shape, dtype=w.dtype, device=w.device)
    mask = torch.tensor([15], dtype=w.dtype, device=w.device)
    for i in range(8):
        shift = 4 * (7 - i)
        result[..., i] = (w >> shift) & mask
    result = result.view(w.shape[0], -1)
    return result


# ## test i4x8_to_i32x8
# value = 1636164468
# print(hex(value))
# a = torch.tensor([[value, value], [value, value]], dtype=torch.int32)
# b = i4x8_to_i32x8(a)
# print(b)
# c = i32x8_to_i4x8(b)
# print(c)
# cmp = a == c
# assert torch.sum(cmp) == cmp.numel()
# exit(0)
# ## end test


def makeup_qweight(in_features: int, out_features: int):
    assert out_features % 8 == 0
    qweight = torch.randint(0,
                            16, (in_features, out_features // 8, 8),
                            dtype=torch.int32,
                            device='cuda')
    print(f'-- makeup qweight: shape {qweight.shape}')
    print(qweight.view(in_features, -1))
    qweight = i32x8_to_i4x8(qweight)
    print(f'-- merge qweight: shape {qweight.shape}')
    print(qweight)
    return qweight


def makup_qzeros(in_features: int, out_features: int, group_size: int):
    assert out_features % 8 == 0
    assert in_features % group_size == 0 and in_features // group_size >= 1

    qzeros = torch.randint(0,
                           16,
                           (in_features // group_size, out_features // 8, 8),
                           dtype=torch.int32,
                           device='cuda')
    print(f'-- makeup qzero: shape {qzeros.shape}')
    print(qzeros.view(in_features // group_size, -1))
    qzeros = i32x8_to_i4x8(qzeros)
    print(f'-- merge qzero: shape {qzeros.shape}\n{qzeros}')
    return qzeros


def makup_scales(in_features: int, out_features: int, group_size: int):
    assert in_features % group_size == 0 and in_features // group_size >= 1
    scales = torch.rand((in_features // group_size, out_features),
                        dtype=torch.float16,
                        device='cuda')
    print(f'-- makeup scales: shape {scales.shape}\n{scales}')
    return scales


def dequantize(qweight, qzeros, scales, group_size: int = 128):
    _qweight = i4x8_to_i32x8(qweight)
    _qzeros = i4x8_to_i32x8(qzeros)
    _qzeros = _qzeros.half()
    weight = _qweight.clone().half()
    for i in range(qzeros.shape[0]):
        start = i * group_size
        end = start + group_size
        weight[start:end] = (weight[start:end, :] -
                             _qzeros[i:i + 1, :]) * scales[i:i + 1, :]
    return weight


# in_features = 128
# out_features = 8
# group_size = 128
# qweight = makeup_qweight(in_features, out_features)
# qzeros = makup_qzeros(in_features=in_features,
#                       out_features=out_features,
#                       group_size=group_size)
# scales = makup_scales(in_features,
#                       out_features=out_features,
#                       group_size=group_size)


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
