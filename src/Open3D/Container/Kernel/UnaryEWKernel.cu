// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Container/Tensor.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Container/CudaUtils.h"

static constexpr int threads_per_block = 128;
static constexpr int items_per_thread = 4;

namespace open3d {
namespace kernel {

template <int threads_per_block, int items_per_thread, typename func_t>
__global__ void elementwise_kernel(int N, func_t f) {
    int items_per_block = threads_per_block * items_per_thread;
    int index = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        if (index < N) {
            f(index);
            index += threads_per_block;
        }
    }
}

static void CopyToContiguousCUDASameDevice(const Tensor& src, Tensor& dst) {
    int N = static_cast<int>(src.GetShape().NumElements());
    int items_per_block = threads_per_block * items_per_thread;
    int grid_size = (N + items_per_block - 1) / items_per_block;
    auto f = [=]OPEN3D_HOST_DEVICE(int idx) {};
    elementwise_kernel<threads_per_block, items_per_thread>
            <<<grid_size, threads_per_block, 0>>>(N, f);
}

void CopyCUDAKernel(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same shape, dtype, and dst
    // - dst must be contiguous
    // - at least one of src or dst is CUDA device

    SizeVector shape = src.GetShape();
    Dtype dtype = src.GetDtype();
    if (src.IsContiguous()) {
        MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                              src.GetDataPtr(), src.GetDevice(),
                              DtypeUtil::ByteSize(dtype) * shape.NumElements());
        utility::LogWarning("Contiguous optimized for {} -> {}\n",
                            src.GetDevice().ToString(),
                            dst.GetDevice().ToString());
    } else {
        if (src.GetDevice() == dst.GetDevice()) {
            CopyToContiguousCUDASameDevice(src, dst);
        } else {
            // Works for both CPU -> GPU or GPU -> CPU
            Tensor src_conti = src.Copy(src.GetDevice());
            // Careful about the resursions
            CopyCUDAKernel(src_conti, dst);
        }
    }
}

}  // namespace kernel
}  // namespace open3d
