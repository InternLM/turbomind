# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import sys

import torch

import turbomind

from .utils import get_u4_slices, pack_u4_row, unpack_awq_gemm

turbomind_dir = osp.split(turbomind.__file__)[0]
sys.path.append(osp.join(turbomind_dir, 'lib'))

try:
    import turbomind_kernels
    TURBOMIND_KERNELS_INSTALLED = True
except Exception as e:
    logging.error(f'turbomind_kernels is not installed: {e}')
    TURBOMIND_KERNELS_INSTALLED = False


def transpose(x):
    return x.t() if x is not None else x


def pad_out_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(-1)
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, pad), 'constant', 0)


def pad_in_dims(x: torch.Tensor, dims: int):
    pad = dims - x.size(0)
    assert x.dim() == 2
    assert pad >= 0
    return torch.nn.functional.pad(x, (0, 0, 0, pad), 'constant', 0)


def to_cuda(x: torch.Tensor, *args):
    return x.cuda()


class Linear(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias: bool = False,
                 quant_method: str = '',
                 w_bit: int = 4,
                 group_size: int = 128,
                 device: str = 'cuda'):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError(
                f'Only 4-bit is supported for now, but got {w_bit} bit')
        if group_size != 128:
            raise NotImplementedError(
                f'Only group_size 128 is supported for now, '
                f'but got group_size {group_size}')
        if bias:
            raise NotImplementedError('bias has not been supported yet')
        self.w_bit = w_bit
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.quant_method = quant_method

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert self.out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            'qweight',
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            'qzeros',
            torch.zeros(
                (in_features // self.group_size, out_features //
                 (32 // self.w_bit)),
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            'scales',
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=device,
            ),
        )

        # if bias:
        #     self.register_buffer(
        #         'bias',
        #         torch.zeros(
        #             (out_features),
        #             dtype=torch.float16,
        #             device=device,
        #         ),
        #     )
        # else:
        #     self.bias = None

        self.linear = turbomind_kernels.Linear(self.in_features,
                                               self.out_features, self.w_bit,
                                               self.group_size)

    def post_init(self):
        assert self.qweight.device.type == 'cuda'
        if self.quant_method == 'awq':
            self.qweight = unpack_awq_gemm(self.qweight).t()
            self.qzeros = unpack_awq_gemm(self.qzeros).t()
            self.scales = self.scales.t()
        elif self.quant_method == 'gptq':
            xs = get_u4_slices(self.qweight, torch.uint8)
            self.qweight = torch.stack(xs, dim=1).view(-1,
                                                       self.qweight.size(-1))
            xs = get_u4_slices(self.qzeros, torch.uint8)
            self.qzeros = torch.stack(xs, dim=-1).view(self.qzeros.size(0),
                                                       -1) + 1
            self.qweight = self.qweight.t()
            self.qzeros = self.qzeros.t()
            self.scales = self.scales.t()
        else:
            return

        self.qweight = transpose(self.qweight)
        self.qzeros = transpose(self.qzeros)
        self.scales = transpose(self.scales)

        self.qweight = pack_u4_row(self.qweight)
        self.qzeros = self.qzeros.to(torch.half)

        device_id = self.qweight.device.index
        properties = torch.cuda.get_device_properties(device_id)

        def is_16xx_series(name):
            import re
            pattern = r'GTX 16\d\d'
            return bool(re.search(pattern, name))

        simt = is_16xx_series(properties.name)
        self.qweight = self.qweight.contiguous()
        self.scales = self.scales.contiguous()
        self.qzeros = self.qzeros.contiguous()
        self.linear.post_init(self.qweight, self.scales, self.qzeros, simt)

    @torch.no_grad()
    def forward(self, x):
        assert TURBOMIND_KERNELS_INSTALLED, (
            'turbomind kernels are not installed. '
            'Please perform `pip install turbomind` to install turbomind '
            'kernels.')
        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        x = x.view(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (self.out_features, )
        out = torch.empty(
            (x.shape[0], self.out_features),
            dtype=torch.float16,
            device=x.device,
        )
        self.linear.forward(x, out)
        out = torch.from_dlpack(out)
        print(out)
        return out.view(out_shape)

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def to_half(x: torch.Tensor):
        return x.to(torch.half)
