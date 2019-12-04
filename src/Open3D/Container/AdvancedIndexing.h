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

#include <vector>

#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

/// \brief Fill dimensions and get output shape for Advanced (fancy) indexing.
/// \param tensor The Tensor to be indexed.
/// \param index_tensors The Tensors that specify the index.
/// \return A tuple of full_index_tensors and output_shape.
std::tuple<std::vector<Tensor>, SizeVector> PreprocessIndexTensors(
        const Tensor& tensor, const std::vector<Tensor>& index_tensors);

class AdvancedIndexing {
public:
    AdvancedIndexing(const Tensor& tensor,
                     const std::vector<Tensor>& index_tensors)
        : tensor_(tensor), index_tensors_(index_tensors) {
        // The constructor makes shallow copies of the tensors to keep input
        // tensors untouched by the preprocessing.
        RunPreprocess();
    }

    Tensor GetPreprocessedTensor() const { return tensor_; }

    std::vector<Tensor> GetPreprocessedIndexTensors() const {
        return index_tensors_;
    }

    SizeVector GetOutputShape() const { return output_shape_; }

    /// Returns true if the indexed dimension is splitted by (full) slice.
    /// E.g. A[[1, 2], :, [1, 2]] returns true
    ///      A[[1, 2], [1, 2], :] returns false
    static bool IsIndexSplittedBySlice(
            const std::vector<Tensor>& index_tensors);

    /// Shuffle indexed dimensions in front of the slice dimensions for the
    /// tensor and index tensors.
    static std::pair<Tensor, std::vector<Tensor>> ShuffleIndexedDimsToFront(
            const Tensor& tensor, const std::vector<Tensor>& index_tensors);

protected:
    /// Preprocess tensor and index tensors.
    void RunPreprocess();

    /// The processed tensors being indexed. The tensor still uses the same
    /// underlying memory, but it may have been reshaped and restrided.
    Tensor tensor_;

    /// The processed index tensors.
    std::vector<Tensor> index_tensors_;

    /// Output shape.
    SizeVector output_shape_;

    // /// Number of dimension actually being indexed.
    // /// E.g. A[[1, 2], :, [1, 2]] returns 2.
    // int64_t ndims_indexed = 0;

    // /// Number of dimension actually being sliced.
    // /// E.g. A[[1, 2], :, [1, 2]] returns 1.
    // int64_t ndims_sliced = 0;
};

}  // namespace open3d
