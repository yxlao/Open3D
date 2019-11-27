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

#include "Open3D/Container/CudaUtils.cuh"
#include "Open3D/Container/Dtype.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

static constexpr int64_t MAX_DIMS = 10;
static constexpr int64_t MAX_OPERANDS = 10;

/// A minimalistic class that reference a Tensor. This class can be used in both
/// host and device code.
///
/// TensorRef is similar to dlpack's DLTensor, with the following differences:
/// - DLTensor stores the full blob's data ptr and init byte_offsets, while
///   TensorRef stores the offseted initial data ptr directly.
/// - TensorRef does not store the device contexts.
/// - TensorRef uses fix-sized array e.g. int64_t shape_[MAX_DIMS], instead of
///   int64_t*.
/// - In the future, we may revisit this part when we enable dlpack support.
struct TensorRef {
    TensorRef(const Tensor& t) {
        data_ptr_ = const_cast<void*>(t.GetDataPtr());
        num_dims_ = t.NumDims();
        dtype_byte_size_ = DtypeUtil::ByteSize(t.GetDtype());
        for (int64_t i = 0; i < num_dims_; ++i) {
            shape_[i] = t.GetShape(i);
            strides_[i] = t.GetStride(i);
        }
    }

    TensorRef() : data_ptr_(nullptr), num_dims_(0), dtype_byte_size_(0) {}

    void* data_ptr_;
    int64_t num_dims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t strides_[MAX_DIMS];
};

/// Indexing Engine for unary-elementwise, binary-elementwise and reduction ops
/// with support for broadcasting.
///
/// After constructing IndexingEngine on the host, the indexing methods can be
/// used from both host and device
class IndexingEngine {
public:
    /// Only single output is supported for simplicity. To extend this funciton
    /// to support multiple outputs, one may check for shape compatibility of
    /// all outputs.
    IndexingEngine(const std::vector<Tensor>& input_tensors,
                   const Tensor& output_tensor,
                   bool enable_reduction = false) {}

    /// Return the total number of workloads (e.g. computations) needed for
    /// the op. The scheduler schedules these workloads to run on parallel
    /// threads.
    ///
    /// Typically for non-reduction ops, NumWorkloads() is the same as
    /// number of output elements.
    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const {
        // TODO
        return 0;
    }

    OPEN3D_HOST_DEVICE char** GetInputPtrs(int64_t workload_idx) const {
        // TODO
        return nullptr;
    }

    OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        // TODO
        return nullptr;
    }

    OPEN3D_HOST_DEVICE int64_t NumInputs() const { return num_inputs_; }

protected:
    /// Whether the op is a reduction op.
    bool enable_reduction_ = false;

    /// Number of input Tensors.
    int64_t num_inputs_ = 0;

    /// Array of input TensorRefs
    TensorRef inputs_[MAX_OPERANDS];

    /// Output TensorRef
    TensorRef output_;
};

}  // namespace open3d
