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

#include "Open3D/Container/MemoryManager.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Open3D/Container/CUDAState.cuh"
#include "Open3D/Container/CUDAUtils.h"

namespace open3d {

CUDAMemoryManager::CUDAMemoryManager() {}

void* CUDAMemoryManager::Malloc(size_t byte_size, const Device& device) {
    void* ptr;
    if (device.GetType() == Device::DeviceType::CUDA) {
        CUDASwitchDeviceInScope switcher(device.GetID());
        OPEN3D_CUDA_CHECK(cudaMalloc(static_cast<void**>(&ptr), byte_size));
    } else {
        utility::LogError("Unimplemented device");
    }
    return ptr;
}

void CUDAMemoryManager::Free(void* ptr, const Device& device) {
    if (device.GetType() == Device::DeviceType::CUDA) {
        if (IsCUDAPointer(ptr) && ptr) {
            CUDASwitchDeviceInScope switcher(device.GetID());
            OPEN3D_CUDA_CHECK(cudaFree(ptr));
        }
    } else {
        utility::LogError("Unimplemented device");
    }
}

void CUDAMemoryManager::Memcpy(void* dst_ptr,
                               const Device& dst_device,
                               const void* src_ptr,
                               const Device& src_device,
                               size_t num_bytes) {
    cudaMemcpyKind memcpy_kind;

    if (dst_device.GetType() == Device::DeviceType::CUDA &&
        src_device.GetType() == Device::DeviceType::CPU) {
        memcpy_kind = cudaMemcpyHostToDevice;
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
    } else if (dst_device.GetType() == Device::DeviceType::CPU &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        memcpy_kind = cudaMemcpyDeviceToHost;
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }
    } else if (dst_device.GetType() == Device::DeviceType::CUDA &&
               src_device.GetType() == Device::DeviceType::CUDA) {
        memcpy_kind = cudaMemcpyDeviceToDevice;
        if (!IsCUDAPointer(dst_ptr)) {
            utility::LogError("dst_ptr is not a CUDA pointer");
        }
        if (!IsCUDAPointer(src_ptr)) {
            utility::LogError("src_ptr is not a CUDA pointer");
        }
    } else {
        utility::LogError("Wrong cudaMemcpyKind");
    }

    CUDASwitchDeviceInScope switcher(src_device.GetID());
    OPEN3D_CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, num_bytes, memcpy_kind));
}

bool CUDAMemoryManager::IsCUDAPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.devicePointer != nullptr) {
        return true;
    }
    return false;
}

}  // namespace open3d
