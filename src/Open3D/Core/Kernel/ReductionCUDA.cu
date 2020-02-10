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

#include "Open3D/Core/Kernel/Reduction.h"

#include "Open3D/Core/CUDAState.cuh"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Kernel/CUDALauncher.cuh"
#include "Open3D/Core/Tensor.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static OPEN3D_HOST_DEVICE void CUDASumReductionKernel(const void* src,
                                                      void* dst) {
    *static_cast<scalar_t*>(dst) += *static_cast<const scalar_t*>(src);
}

void ReductionCUDA(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keep_dim,
                   ReductionOpCode op_code) {
    Dtype dtype = dst.GetDtype();
    Indexer indexer({src}, dst, DtypePolicy::ASSERT_SAME, dims);

    CUDADeviceSwitcher switcher(src.GetDevice());
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        // Optain identity and element kernel based on op_code.
        scalar_t identity;
        std::function<void(void*, void*)> element_kernel;
        switch (op_code) {
            case ReductionOpCode::Sum:
                identity = static_cast<scalar_t>(0);
                element_kernel = CUDASumReductionKernel<scalar_t>;
                break;
            default:
                utility::LogError("Unsupported op code.");
                break;
        }
        dst.Fill(identity);
    });

    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        cuda_launcher::LaunchReductionKernelOneOutput<scalar_t>(
                indexer,
                // Need to wrap as extended CUDA lamba function
                [] OPEN3D_HOST_DEVICE(const void* src, void* dst) {
                    CUDASumReductionKernel<scalar_t>(src, dst);
                });
    });

    if (dims.size() != src.NumDims()) {
        utility::LogError("Unimplemented case for ReductionCUDA.");
    }

    // utility::LogError("Unimplemented ReductionCUDA.");
}

}  // namespace kernel
}  // namespace open3d
