#!/bin/bash
WORKSPACE_PATH=$(dirname "$(readlink -f "$0")")

builder="-G Ninja"

if [ "$1" == "make" ]; then
    builder=""
fi

cmake ${builder} .. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_CUDA_FLAGS="-lineinfo" \
    -DUSE_NVTX=ON \
    -DBUILD_TEST=ON \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DLMDEPLOY_ASAN_ENABLE=OFF \
    -DLMDEPLOY_UBSAN_ENABLE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80-real"
