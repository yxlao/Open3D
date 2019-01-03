//
// Created by wei on 12/31/18.
//

#pragma once

#include <Cuda/Common/Common.h>
#include <Eigen/Eigen>

#include <cassert>

namespace open3d {
namespace cuda {
/**
 * M x N matrix
 */
template<typename T, size_t M, size_t N>
class MatrixCuda {
public:
    T v[M * N];

public:
    __HOSTDEVICE__ MatrixCuda() {}

    __HOSTDEVICE__ T& operator() (size_t i, size_t j) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < M && j < N);
#endif
        return v[i * N + j];
    }
};
}
}