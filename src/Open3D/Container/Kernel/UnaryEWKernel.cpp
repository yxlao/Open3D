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

#include "Open3D/Container/Kernel/UnaryEW.h"

#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

void CopyCPUKernel(const Tensor& src, Tensor& dst) {
    // src and dst have been checked to have the same shape, dtype, device, and
    // dst must be contiguous
    SizeVector shape = src.GetShape();
    SizeVector strides = src.GetStrides();
    Dtype dtype = src.GetDtype();
    Device device = src.GetDevice();

    int64_t num_elements = static_cast<int64_t>(shape.NumElements());
    size_t num_dims = shape.size();
    SizeVector default_strides = Tensor::DefaultStrides(shape);
    size_t element_byte_size = DtypeUtil::ByteSize(dtype);

    const uint8_t* src_data_ptr = static_cast<const uint8_t*>(src.GetDataPtr());
    uint8_t* dst_data_ptr = static_cast<uint8_t*>(dst.GetDataPtr());

    if (src.IsContiguous()) {
        MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                              src.GetDataPtr(), src.GetDevice(),
                              element_byte_size * num_elements);
        utility::LogWarning("Contiguous optimized for cpu -> cpu\n");
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        // int64_t to avoid MSVC openmp error
        // TODO: Benchmark auto-vectorization v.s. OpenMP
        for (int64_t dst_offset = 0; dst_offset < num_elements; dst_offset++) {
            size_t ind = static_cast<size_t>(dst_offset);
            SizeVector indices(shape.size());
            size_t src_offset = 0;
            for (size_t dim = 0; dim < num_dims; dim++) {
                src_offset += ind / default_strides[dim] * strides[dim];
                ind = ind % default_strides[dim];
            }
            const void* src_ptr = src_data_ptr + src_offset * element_byte_size;
            void* dst_ptr = dst_data_ptr + dst_offset * element_byte_size;
            MemoryManager::Memcpy(dst_ptr, device,
                                  const_cast<const void*>(src_ptr), device,
                                  element_byte_size);
        }
    }
}

}  // namespace kernel
}  // namespace open3d
