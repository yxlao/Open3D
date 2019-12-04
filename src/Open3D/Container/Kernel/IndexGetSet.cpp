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

#include "Open3D/Container/Kernel/IndexGetSet.h"

#include "Open3D/Container/Dispatch.h"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/Kernel/CPULauncher.h"
#include "Open3D/Container/MemoryManager.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static void CPUCopyElementKernel(const void* src, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(src);
}

void IndexGetCPU(const Tensor& src,
                 Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const SizeVector& indexed_out_shape) {
    Dtype dtype = src.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        CPULauncher::LaunchRhsIndexedUnaryEWKernel<scalar_t>(
                src, dst, index_tensors, indexed_out_shape,
                CPUCopyElementKernel<scalar_t>);
    });
}

void IndexSetCPU(const Tensor& src,
                 Tensor& dst,
                 const std::vector<Tensor>& index_tensors,
                 const SizeVector& indexed_out_shape) {
    Dtype dtype = src.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        CPULauncher::LaunchLhsIndexedUnaryEWKernel<scalar_t>(
                src, dst, index_tensors, indexed_out_shape,
                CPUCopyElementKernel<scalar_t>);
    });
}
}  // namespace kernel
}  // namespace open3d
