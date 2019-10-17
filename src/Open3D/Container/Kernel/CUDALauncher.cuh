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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

static constexpr int MAX_DIMS = 10;

namespace open3d {
namespace kernel {

class OffsetCalculator {
public:
    OffsetCalculator(size_t num_dims,
                     const size_t* src_strides,
                     const size_t* dst_strides)
        : num_dims_(num_dims) {
#pragma unroll
        for (size_t i = 0; i < num_dims_; i++) {
            src_strides_[i] = src_strides[i];
            dst_strides_[i] = dst_strides[i];
        }
    }

    OPEN3D_HOST_DEVICE size_t GetOffset(size_t idx) const {
        size_t src_idx = 0;
#pragma unroll
        for (size_t dim = 0; dim < num_dims_; dim++) {
            src_idx += idx / dst_strides_[dim] * src_strides_[dim];
            idx = idx % dst_strides_[dim];
        }
        return src_idx;
    }

protected:
    size_t num_dims_;
    size_t src_strides_[MAX_DIMS];
    size_t dst_strides_[MAX_DIMS];
};

}  // namespace kernel
}  // namespace open3d
