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

bool IsCompatibleBroadcastShape(const SizeVector& l_shape,
                                const SizeVector& r_shape) {
    int64_t l_ndim = l_shape.size();
    int64_t r_ndim = r_shape.size();

    if (l_ndim == 0 || r_ndim == 0) {
        return true;
    }

    // Only need to check the last `shorter_ndim` dims
    // E.g. LHS: [100, 200, 2, 3, 4]
    //      RHS:           [2, 1, 4] <- only last 3 dims need to be checked
    // Checked from right to left
    int64_t shorter_ndim = std::min(l_ndim, r_ndim);
    for (int64_t ind = 0; ind < shorter_ndim; ++ind) {
        int64_t l_dim = l_shape[l_ndim - 1 - ind];
        int64_t r_dim = r_shape[r_ndim - 1 - ind];
        if (!(l_dim == r_dim || l_dim == 1 || r_dim == 1)) {
            return false;
        }
    }
    return true;
}

SizeVector BroadcastedShape(const SizeVector& l_shape,
                            const SizeVector& r_shape) {
    if (!IsCompatibleBroadcastShape(l_shape, r_shape)) {
        utility::LogError("Shape {} and {} are not broadcast-compatible",
                          l_shape, r_shape);
    }

    int64_t l_ndim = l_shape.size();
    int64_t r_ndim = r_shape.size();
    int64_t shorter_ndim = std::min(l_ndim, r_ndim);
    int64_t longer_ndim = std::max(l_ndim, r_ndim);

    if (l_ndim == 0) {
        return r_shape;
    }
    if (r_ndim == 0) {
        return l_shape;
    }

    SizeVector broadcasted_shape(longer_ndim, 0);
    // Checked from right to left
    for (int64_t ind = 0; ind < longer_ndim; ind++) {
        int64_t l_ind = l_ndim - longer_ndim + ind;
        int64_t r_ind = r_ndim - longer_ndim + ind;
        int64_t l_dim = l_ind >= 0 ? l_shape[l_ind] : 0;
        int64_t r_dim = r_ind >= 0 ? r_shape[r_ind] : 0;
        broadcasted_shape[ind] = std::max(l_dim, r_dim);
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
