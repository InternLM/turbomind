// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <istream>
#include <ostream>
#include <memory>
#include <cuda_runtime.h>
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/utils/tensor.h"

namespace turbomind {


enum class WeightType : int
{
    kFP32,
    kFP16,
    kFP8,  // not supported yet
    kBF16,
    kINT8,
    kINT4
};

std::shared_ptr<Tensor> convert_qweight(std::shared_ptr<Tensor> qweight,
                                        size_t input_dims,
                                        size_t output_dims,
                                        bool use_simt);
std::shared_ptr<Tensor> convert_scales_zeros(std::shared_ptr<Tensor> scales,
                                             std::shared_ptr<Tensor> qzeros,
                                             std::shared_ptr<Tensor> scales_zeros,
                                             size_t input_dims,
                                             size_t output_dims,
                                             int group_size,
                                             bool use_simt);


class Linear {
public:
    Linear(size_t input_dims, size_t output_dims, int w_bit, int group_size);
    void post_init(std::shared_ptr<Tensor> qweight, std::shared_ptr<Tensor> scales, std::shared_ptr<Tensor> qzeros,
                   bool simt);
    void forward(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out);
    ~Linear() {}

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
};
