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

#include "Open3D/Container/Broadcast.h"

#include "Open3D/Container/SizeVector.h"
#include "Open3D/Container/Tensor.h"

namespace open3d {

bool IsCompatibleBroadcastShape(const SizeVector& lhs_shape,
                                const SizeVector& rhs_shape) {
    int64_t left_ndim = lhs_shape.size();
    int64_t right_ndim = rhs_shape.size();

    if (left_ndim == 0 || right_ndim == 0) {
        return true;
    }

    // Only need to check the last `shorter_ndim` dims
    // E.g. LHS: [100, 200, 2, 3, 4]
    //      RHS:           [2, 1, 4] <- only last 3 dims need to be checked
    // Checked from right to left
    int64_t shorter_ndim = std::min(left_ndim, right_ndim);
    for (int64_t ind = 0; ind < shorter_ndim; ++ind) {
        int64_t left_dim = lhs_shape[left_ndim - 1 - ind];
        int64_t right_dim = rhs_shape[right_ndim - 1 - ind];
        if (!(left_dim == right_dim || left_dim == 1 || right_dim == 1)) {
            return false;
        }
    }
    return true;
}

SizeVector BroadcastedShape(const SizeVector& lhs_shape,
                            const SizeVector& rhs_shape) {
    if (!IsCompatibleBroadcastShape(lhs_shape, rhs_shape)) {
        utility::LogError("Shape {} and {} are not broadcast-compatible",
                          lhs_shape, rhs_shape);
    }

    int64_t left_ndim = lhs_shape.size();
    int64_t right_ndim = rhs_shape.size();
    int64_t shorter_ndim = std::min(left_ndim, right_ndim);
    int64_t longer_ndim = std::max(left_ndim, right_ndim);

    if (left_ndim == 0) {
        return rhs_shape;
    }
    if (right_ndim == 0) {
        return lhs_shape;
    }

    SizeVector broadcasted_shape(longer_ndim, 0);
    // Checked from right to left
    for (int64_t ind = 0; ind < longer_ndim; ind++) {
        int64_t left_ind = left_ndim - longer_ndim + ind;
        int64_t right_ind = right_ndim - longer_ndim + ind;
        int64_t left_dim = left_ind >= 0 ? lhs_shape[left_ind] : 0;
        int64_t right_dim = right_ind >= 0 ? rhs_shape[right_ind] : 0;
        broadcasted_shape[ind] = std::max(left_dim, right_dim);
    }
    return broadcasted_shape;
}

bool CanBeBrocastedToShape(const SizeVector& src_shape,
                           const SizeVector& dst_shape) {
    if (IsCompatibleBroadcastShape(src_shape, dst_shape)) {
        return BroadcastedShape(src_shape, dst_shape) == dst_shape;
    } else {
        return false;
    }
}

}  // namespace open3d
