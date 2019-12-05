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

#include "Open3D/Container/AdvancedIndexer.h"

#include "Open3D/Container/Broadcast.h"
#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

bool AdvancedIndexer::IsIndexSplittedBySlice(
        const std::vector<Tensor>& index_tensors) {
    bool index_dim_started = false;
    bool index_dim_ended = false;
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() == 0) {
            // This dimension is sliced.
            if (index_dim_started) {
                index_dim_ended = true;
            }
        } else {
            // This dimension is indexed.
            if (index_dim_ended) {
                return true;
            }
            if (!index_dim_started) {
                index_dim_started = true;
            }
        }
    }
    return false;
}

std::pair<Tensor, std::vector<Tensor>>
AdvancedIndexer::ShuffleIndexedDimsToFront(
        const Tensor& tensor, const std::vector<Tensor>& index_tensors) {
    int64_t ndims = tensor.NumDims();
    std::vector<int64_t> permutation;
    std::vector<Tensor> permuted_index_tensors;
    for (int64_t i = 0; i < ndims; ++i) {
        if (index_tensors[i].NumDims() != 0) {
            permutation.push_back(i);
            permuted_index_tensors.emplace_back(index_tensors[i]);
        }
    }
    for (int64_t i = 0; i < ndims; ++i) {
        if (index_tensors[i].NumDims() == 0) {
            permutation.push_back(i);
            permuted_index_tensors.emplace_back(index_tensors[i]);
        }
    }
    return std::make_pair(tensor.Permute(permutation),
                          std::move(permuted_index_tensors));
}

std::pair<std::vector<Tensor>, SizeVector>
AdvancedIndexer::ExpandToCommonShapeExcpetZeroDim(
        const std::vector<Tensor>& index_tensors) {
    SizeVector replacement_shape({});  // {} can be broadcasted to any shape.
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() != 0) {
            replacement_shape = BroadcastedShape(replacement_shape,
                                                 index_tensor.GetShape());
        }
    }

    std::vector<Tensor> expanded_tensors;
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.NumDims() == 0) {
            expanded_tensors.push_back(index_tensor);
        } else {
            expanded_tensors.push_back(index_tensor.Expand(replacement_shape));
        }
    }

    return std::make_pair(expanded_tensors, replacement_shape);
}

Tensor AdvancedIndexer::RestrideTensor(const Tensor& tensor,
                                       int64_t dims_before,
                                       int64_t dims_indexed,
                                       SizeVector replacement_shape) {
    SizeVector shape = tensor.GetShape();
    SizeVector strides = tensor.GetStrides();
    int64_t end = dims_before + dims_indexed;
    shape.erase(shape.begin() + dims_before, shape.begin() + end);
    strides.erase(strides.begin() + dims_before, strides.begin() + end);
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(),
                 replacement_shape.end());
    strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
    return tensor.AsStrided(shape, strides);
}

Tensor AdvancedIndexer::RestrideIndexTensor(const Tensor& index_tensor,
                                            int64_t dims_before,
                                            int64_t dims_after) {
    SizeVector old_shape = index_tensor.GetShape();
    SizeVector new_shape(dims_before + index_tensor.NumDims() + dims_after, 1);
    std::copy(old_shape.begin(), old_shape.end(),
              new_shape.begin() + dims_before);
    Tensor reshaped = index_tensor.Reshape(new_shape);
    return reshaped;
}

void AdvancedIndexer::RunPreprocess() {
    // Dimension check
    if (index_tensors_.size() > tensor_.NumDims()) {
        utility::LogError(
                "Number of index_tensors {} exceeds tensor dimension "
                "{}.",
                index_tensors_.size(), tensor_.NumDims());
    }

    // Index tensors must be using int64.
    // Boolean indexing tensors will be supported in the future by
    // converting to int64_t tensors.
    for (const Tensor& index_tensor : index_tensors_) {
        if (index_tensor.GetDtype() != Dtype::Int64) {
            utility::LogError(
                    "Index tensor must have Int64 dtype, but {} was used.",
                    DtypeUtil::ToString(index_tensor.GetDtype()));
        }
    }

    // Fill implied 0-d index tensors at the tail dimensions.
    // 0-d index tensor represents a fully sliced dimension, i.e. [:] in Numpy.
    // Partial slice e.g. [1:3] shall be handled outside of advanced indexing.
    //
    // E.g. Given A.shape == [5, 6, 7, 8],
    //      A[[1, 2], [3, 4]] is converted to
    //      A[[1, 2], [3, 4], :, :].
    Tensor empty_index_tensor =
            Tensor(SizeVector(), Dtype::Int64, tensor_.GetDevice());
    int64_t num_omitted_dims = tensor_.NumDims() - index_tensors_.size();
    for (int64_t i = 0; i < num_omitted_dims; ++i) {
        index_tensors_.push_back(empty_index_tensor);
    }

    // Fill 0 to 0-d index tensors. The omitted indexing tensors is equivalent
    // to always increment offset 0.
    for (Tensor& index_tensor : index_tensors_) {
        if (index_tensor.NumDims() == 0) {
            index_tensor.Fill(0);
        }
    }

    // Transpose all indexed dimensions to front if indexed dimensions are
    // splitted by sliced dimensions. The tensor being indexed are dimshuffled
    // accordingly.
    //
    // E.g. Given A.shape == [5, 6, 7, 8],
    //      A[[1, 2], :, [3, 4], :] is converted to
    //      A.permute([0, 2, 1, 3])[[1, 2], [3, 4], :, :].
    //      The resulting shape is (2, 6, 8).
    //
    // See "Combining advanced and basic indexing" section of
    // https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    if (IsIndexSplittedBySlice(index_tensors_)) {
        std::tie(tensor_, index_tensors_) =
                ShuffleIndexedDimsToFront(tensor_, index_tensors_);
    }

    // Put index tensors_ on the same device as tensor_.
    for (size_t i = 0; i < index_tensors_.size(); ++i) {
        if (index_tensors_[i].GetDevice() != tensor_.GetDevice()) {
            index_tensors_[i] = index_tensors_[i].Copy(tensor_.GetDevice());
        }
    }

    // Expand (broadcast with view) all index_tensors_ to a common shape,
    // ignoring 0-d index_tensors_.
    SizeVector replacement_shape;
    std::tie(index_tensors_, replacement_shape) =
            ExpandToCommonShapeExcpetZeroDim(index_tensors_);

    int64_t dims_before = 0;
    int64_t dims_after = 0;
    int64_t dims_indexed = 0;
    bool replacement_shape_inserted = false;
    for (size_t dim = 0; dim < index_tensors_.size(); dim++) {
        if (index_tensors_[dim].NumDims() == 0) {
            if (dims_indexed == 0) {
                dims_before++;
            } else {
                dims_after++;
            }
            output_shape_.push_back(tensor_.GetShape(dim));
        } else {
            if (!replacement_shape_inserted) {
                output_shape_.insert(output_shape_.end(),
                                     replacement_shape.begin(),
                                     replacement_shape.end());
                replacement_shape_inserted = true;
            }
            dims_indexed++;
            indexed_shape_.push_back(tensor_.GetShape(dim));
            indexed_strides_.push_back(tensor_.GetStride(dim));
        }
    }

    // If the indexed_shape_ contains a dimension of size 0 but the
    // replacement shape does not, the index is out of bounds. This is because
    // there is no valid number to index an empty tensor.
    // Normally, out of bounds is detected in the advanded indexing kernel. We
    // detecte here for more helpful error message.
    auto contains_zero = [](const SizeVector& vals) -> bool {
        return std::any_of(vals.begin(), vals.end(),
                           [](int64_t val) { return val == 0; });
    };
    if (contains_zero(indexed_shape_) && !contains_zero(replacement_shape)) {
        utility::LogError("Index is out of bounds for dimension with size 0");
    }

    // Restride tensor_ and index tensors_.
    tensor_ = RestrideTensor(tensor_, dims_before, dims_indexed,
                             replacement_shape);
    for (size_t dim = 0; dim < index_tensors_.size(); dim++) {
        if (index_tensors_[dim].NumDims() != 0) {
            index_tensors_[dim] = RestrideIndexTensor(index_tensors_[dim],
                                                      dims_before, dims_after);
        }
    }

    utility::LogDebug("tensor_.GetShape().ToString(): {}",
                      tensor_.GetShape().ToString());
    for (const auto& index_tensor : index_tensors_) {
        utility::LogDebug("index_tensor.GetShape().ToString(): {}",
                          index_tensor.GetShape().ToString());
    }
    utility::LogDebug("dims_before {}, dims_after {}, dims_indexed {}",
                      dims_before, dims_after, dims_indexed);
    utility::LogDebug("indexed_shape_ {}", indexed_shape_.ToString());
    utility::LogDebug("indexed_strides_ {}", indexed_strides_.ToString());
    utility::LogDebug("replacement_shape {}", replacement_shape.ToString());
    utility::LogDebug("output_shape_ {}", output_shape_.ToString());
}

std::tuple<std::vector<Tensor>, SizeVector> PreprocessIndexTensors(
        const Tensor& tensor, const std::vector<Tensor>& index_tensors) {
    // Index tensors must be using int64_t
    for (const Tensor& index_tensor : index_tensors) {
        if (index_tensor.GetDtype() != Dtype::Int64) {
            utility::LogError(
                    "Indexing Tensor must have Int64 dtype, but {} was used.",
                    DtypeUtil::ToString(index_tensor.GetDtype()));
        }
    }

    // Fill implied 0-d indexing tensors at the tail dimensions.
    Tensor empty_index_tensor =
            Tensor(SizeVector(), Dtype::Int64, tensor.GetDevice());
    std::vector<Tensor> full_index_tensors = index_tensors;
    for (int64_t i = 0; i < tensor.NumDims() - index_tensors.size(); ++i) {
        full_index_tensors.push_back(empty_index_tensor);
    }

    // Find all trivial and non-trivial index_tensors
    std::vector<int64_t> trivial_dims;
    std::vector<int64_t> non_trivial_dims;
    std::vector<SizeVector> non_trivial_shapes;
    for (int64_t dim = 0; dim < full_index_tensors.size(); ++dim) {
        if (full_index_tensors[dim].NumDims() == 0) {
            trivial_dims.push_back(dim);
        } else {
            non_trivial_dims.push_back(dim);
            non_trivial_shapes.push_back(full_index_tensors[dim].GetShape());
        }
    }

    // Broadcast non-trivial shapes
    SizeVector broadcasted_non_trivial_shape = {};
    for (const SizeVector& non_trivial_shape : non_trivial_shapes) {
        if (IsCompatibleBroadcastShape(broadcasted_non_trivial_shape,
                                       non_trivial_shape)) {
            broadcasted_non_trivial_shape = BroadcastedShape(
                    broadcasted_non_trivial_shape, non_trivial_shape);
        } else {
            utility::LogError(
                    "Index shapes broadcsting error, {} and {} are not "
                    "compatible.",
                    broadcasted_non_trivial_shape, non_trivial_shape);
        }
    }

    if (broadcasted_non_trivial_shape.size() != 1) {
        utility::LogError("Only supporting 1D index tensor for now.");
    }

    // Now, broadcast non-trivial index tensors
    for (int64_t i = 0; i < full_index_tensors.size(); ++i) {
        if (full_index_tensors[i].NumDims() != 0) {
            full_index_tensors[i].Assign(full_index_tensors[i].Broadcast(
                    broadcasted_non_trivial_shape));
        }
    }

    for (int64_t i = 1; i < non_trivial_dims.size(); ++i) {
        if (non_trivial_dims[i - 1] + 1 != non_trivial_dims[i]) {
            utility::LogError(
                    "Only supporting the case where advanced indices are all"
                    "next to each other, however advanced index in dimension "
                    "{} and {} are separated by one or more slices.",
                    non_trivial_dims[i - 1], non_trivial_dims[i]);
        }
    }

    SizeVector output_shape;
    std::vector<int64_t> slice_map;
    bool filled_non_trivial_dims = false;
    const auto& tensor_shape = tensor.GetShape();
    for (int64_t dim = 0; dim < tensor_shape.size(); ++dim) {
        if (full_index_tensors[dim].NumDims() == 0) {
            output_shape.emplace_back(tensor_shape[dim]);
            slice_map.emplace_back(dim);
        } else {
            if (!filled_non_trivial_dims) {
                // broadcasted_non_trivial_shape is 1-D for now
                output_shape.emplace_back(broadcasted_non_trivial_shape[0]);
                filled_non_trivial_dims = true;
                slice_map.emplace_back(-1);
            }
        }
    }

    return std::make_tuple(full_index_tensors, output_shape);
}

}  // namespace open3d
