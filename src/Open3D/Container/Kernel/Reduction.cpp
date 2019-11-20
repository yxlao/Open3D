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

namespace open3d {
namespace kernel {

void Max(const Tensor& src, Tensor& dst) {
    if (dst.GetShape() != SizeVector()) {
        utility::LogError(
                "Reduction kernel only supports scalar output for now");
    }

    if (src.GetDtype() != dst.GetDtype()) {
        utility::LogError("src and dst tensor dtype mismatch {} != {}",
                          DtypeUtil::ToString(src.GetDtype()),
                          DtypeUtil::ToString(dst.GetDtype()));
    }

    // Disbatch to device
    Device::DeviceType src_device_type = src.GetDevice().device_type_;
    Device::DeviceType dst_device_type = dst.GetDevice().device_type_;

    if (src_device_type == Device::DeviceType::CPU &&
        dst_device_type != Device::DeviceType::CPU) {
    }
#ifdef BUILD_CUDA_MODULE
    else if (src_device_type == Device::DeviceType::CUDA &&
             dst_device_type != Device::DeviceType::CUDA) {
        MaxCPU(src, dst);
    } else if (src_device_type == Device::DeviceType::CPU &&
               dst_device_type != Device::DeviceType::CUDA) {
        Tensor dst_on_cpu(dst.GetShape(), dst.GetDtype(), src.GetDevice());
        MaxCPU(src, dst_on_cpu);
        dst.CopyFrom(dst_on_cpu);
    } else if (src_device_type == Device::DeviceType::CUDA &&
               dst_device_type != Device::DeviceType::CPU) {
        Tensor dst_on_cuda(dst.GetShape(), dst.GetDtype(), src.GetDevice());
        MaxCUDA(src, dst_on_cuda);
        dst.CopyFrom(dst_on_cuda);
    }
#endif
    else {
        utility::LogError("Unimplemented src_device {} or dst_device {}",
                          src.GetDevice().ToString(),
                          dst.GetDevice().ToString());
    }
}

}  // namespace kernel
}  // namespace open3d
