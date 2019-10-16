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

#include "Open3D/Container/CudaUtils.cuh"

static constexpr int threads_per_block = 128;
static constexpr int items_per_thread = 4;
constexpr int MAX_DIMS = 25;

namespace open3d {
namespace kernel {

template <int threads_per_block, int items_per_thread, typename func_t>
__global__ void elementwise_kernel(int N, func_t f) {
    int items_per_block = threads_per_block * items_per_thread;
    int idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        if (idx < N) {
            f(idx);
            idx += threads_per_block;
        }
    }
}

template <typename T>
OPEN3D_HOST_DEVICE void templated_copy(const void* src, void* dst) {
    *static_cast<T*>(dst) = *static_cast<const T*>(src);
}

struct OffsetCalculator {
    OffsetCalculator(size_t num_dims,
                     const size_t* src_strides,
                     const size_t* dst_strides)
        : num_dims_(num_dims) {
        for (size_t i = 0; i < num_dims_; i++) {
            src_strides_[i] = src_strides[i];
            dst_strides_[i] = dst_strides[i];
        }
    }

    OPEN3D_HOST_DEVICE size_t GetOffset(size_t idx) const {
        size_t src_idx = 0;
        for (size_t dim = 0; dim < num_dims_; dim++) {
            src_idx += idx / dst_strides_[dim] * src_strides_[dim];
            idx = idx % dst_strides_[dim];
        }
        return src_idx;
    }

    size_t num_dims_;
    size_t src_strides_[MAX_DIMS];
    size_t dst_strides_[MAX_DIMS];
};

template <typename T>
static void CopyToContiguousCUDASameDevice(const Tensor& src, Tensor& dst) {
    int N = static_cast<int>(src.GetShape().NumElements());
    int items_per_block = threads_per_block * items_per_thread;
    int grid_size = (N + items_per_block - 1) / items_per_block;

    const uint8_t* src_data_ptr = static_cast<const uint8_t*>(src.GetDataPtr());
    uint8_t* dst_data_ptr = static_cast<uint8_t*>(dst.GetDataPtr());
    int element_byte_size = DtypeUtil::ByteSize(src.GetDtype());
    OffsetCalculator offset_calculator(src.GetShape().size(),
                                       src.GetStrides().data(),
                                       dst.GetStrides().data());

    auto f = [=] OPEN3D_HOST_DEVICE(int idx) {
        int src_idx = offset_calculator.GetOffset(idx);
        const void* src_ptr = src_data_ptr + src_idx * element_byte_size;
        void* dst_ptr = dst_data_ptr + idx * element_byte_size;
        templated_copy<T>(src_ptr, dst_ptr);
    };

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
            utility::LogWarning("To launch kernel for {} -> {}\n",
                                src.GetDevice().ToString(),
                                dst.GetDevice().ToString());
            switch (dtype) {
                case Dtype::Float32:
                    CopyToContiguousCUDASameDevice<float>(src, dst);
                    break;
                case Dtype::Float64:
                    CopyToContiguousCUDASameDevice<double>(src, dst);
                    break;
                case Dtype::Int32:
                    CopyToContiguousCUDASameDevice<int32_t>(src, dst);
                    break;
                case Dtype::Int64:
                    CopyToContiguousCUDASameDevice<int64_t>(src, dst);
                    break;
                case Dtype::UInt8:
                    CopyToContiguousCUDASameDevice<uint8_t>(src, dst);
                    break;
                default:
                    utility::LogFatal("Unsupported data type\n");
            }
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
