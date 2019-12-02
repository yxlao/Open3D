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

#include "Open3D/Container/Broadcast.h"
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
    TensorRef() : data_ptr_(nullptr), ndims_(0), dtype_byte_size_(0) {}

    TensorRef(const Tensor& t) {
        data_ptr_ = static_cast<char*>(const_cast<void*>(t.GetDataPtr()));
        ndims_ = t.NumDims();
        dtype_byte_size_ = DtypeUtil::ByteSize(t.GetDtype());
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = t.GetShape(i);
            strides_[i] = t.GetStride(i);
        }
    }

    TensorRef(const TensorRef& tr) {
        data_ptr_ = tr.data_ptr_;
        ndims_ = tr.ndims_;
        dtype_byte_size_ = tr.dtype_byte_size_;
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = tr.shape_[i];
            strides_[i] = tr.strides_[i];
        }
    }

    char* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t strides_[MAX_DIMS];
};

/// Indexing Engine for elementwise ops with broadcasting support.
///
/// Fancy indexing is supported by restriding input tensor and treating the
/// operation as elementwise op. Reduction op will be supported by
/// IndexingEngine in the future.
///
/// After constructing IndexingEngine on the host, the indexing methods can be
/// used from both host and device.
class IndexingEngine {
public:
    /// Only single output is supported for simplicity. To extend this function
    /// to support multiple outputs, one may check for shape compatibility of
    /// all outputs.
    IndexingEngine(const std::vector<Tensor>& input_tensors,
                   const Tensor& output_tensor) {
        // Conver to TensorRef.
        for (int64_t i = 0; i < input_tensors.size(); ++i) {
            inputs_[i] = TensorRef(input_tensors[i]);
        }
        output_ = TensorRef(output_tensor);

        // Broadcast inputs to match output shape.
        for (TensorRef& input : inputs_) {
            BroadcastRestride(input, output_);
        }
    }

    /// Broadcast \p src to \p dst by assigning shape 1 to omitted dimensions
    /// and stride 0 to broadcasted dimensions. This allows element-wise
    /// iteration of input and output tensors based on the new shape and
    /// strides.
    ///
    /// [Before]
    /// src.shape_:   [     2,  1,  3]
    /// src.strides_: [     3,  3,  1]
    /// dst.shape_:   [ 2,  2,  2,  3]
    /// dst.strides_: [12,  6,  3,  1]
    /// [After]
    /// src.shape_:   [ 1,  2,  1,  3]  # Updated
    /// src.strides_: [ 0,  3,  0,  1]  # Updated
    /// dst.shape_:   [ 2,  2,  2,  3]  # Unchanged
    /// dst.strides_: [12,  6,  3,  1]  # Unchanged
    ///
    /// \param src The source TensorRef to be broadcasted.
    /// \param dst The destination TensorRef to be broadcasted to.
    static void BroadcastRestride(TensorRef& src, const TensorRef& dst) {
        int64_t src_ndims = src.ndims_;
        int64_t dst_ndims = dst.ndims_;
        int64_t ndims = dst_ndims;

        // Fill omitted dimensions.
        for (int64_t i = 0; i < ndims - src_ndims; ++i) {
            src.shape_[i] = 1;
            src.strides_[i] = 0;
        }

        // Fill broadcasted dimensions.
        for (int64_t i = 0; i < src_ndims; ++i) {
            src.shape_[ndims - src_ndims + i] = src.shape_[i];
            if (src.shape_[i] == 1) {
                src.strides_[ndims - src_ndims + i] = 0;
            } else {
                src.strides_[ndims - src_ndims + i] = src.strides_[i];
            }
        }
    }

    OPEN3D_HOST_DEVICE TensorRef* GetNumWorkloads() { return inputs_; }

    /// Return the total number of workloads (e.g. computations) needed for
    /// the op. The scheduler schedules these workloads to run on parallel
    /// threads.
    ///
    /// Typically for non-reduction ops, NumWorkloads() is the same as
    /// number of output elements.
    OPEN3D_HOST_DEVICE int64_t NumWorkloads() const { return 0; }

    OPEN3D_HOST_DEVICE char** GetInputPtrs(int64_t workload_idx) const {
        // TODO
        return nullptr;
    }

    OPEN3D_HOST_DEVICE char* GetOutputPtr(int64_t workload_idx) const {
        // TODO
        return nullptr;
    }

    OPEN3D_HOST_DEVICE int64_t NumInputs() const { return num_inputs_; }

    OPEN3D_HOST_DEVICE TensorRef* GetInputs() { return inputs_; }

    OPEN3D_HOST_DEVICE TensorRef GetOutput() { return output_; }

protected:
    /// Number of input Tensors.
    int64_t num_inputs_ = 0;

    /// Array of input TensorRefs.
    TensorRef inputs_[MAX_OPERANDS];

    /// Output TensorRef.
    TensorRef output_;
};

}  // namespace open3d
