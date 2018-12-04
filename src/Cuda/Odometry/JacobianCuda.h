//
// Created by wei on 10/4/18.
//

#pragma once

#include <Cuda/Common/VectorCuda.h>
#include <cstdlib>

namespace open3d {

namespace cuda {
/** Approximated by JtJ in optimization **/
template<size_t N>
class HessianCuda {
    /**
     * 0  1  2  3  4  5
     *    6  7  8  9  10
     *       11 12 13 14
     *          15 16 17
     *             18 19
     *                20
     **/
private:
    float h_[(N + 1) * N / 2];

public:
    inline __HOSTDEVICE__ float &operator()(size_t i) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < (N + 1) * N / 2);
#endif
        return h_[i];
    }

    inline __HOSTDEVICE__ const float &operator()(size_t i) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < (N + 1) * N / 2);
#endif
        return h_[i];
    }

    inline __HOSTDEVICE__ HessianCuda<N> operator+(
        const HessianCuda<N> &other) {
        HessianCuda<N> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < (N + 1) * N / 2; ++i) {
            ret(i) = other(i) + h_[i];
        }
        return ret;
    }

    inline __HOSTDEVICE__ HessianCuda<N> &operator+=(
        const HessianCuda<N> &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < (N + 1) * N / 2; ++i) {
            h_[i] += other(i);
        }
        return (*this);
    }
};

template<size_t N>
class JacobianCuda {
private:
    float j_[N];

public:
    inline __HOSTDEVICE__ float &operator()(size_t i) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return j_[i];
    }
    inline __HOSTDEVICE__ const float &operator()(size_t i) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return j_[i];
    }

    inline __HOSTDEVICE__ HessianCuda<N> ComputeJtJ() {
        HessianCuda<N> h;
        int cnt = 0;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
            for (int j = i; j < N; ++j) {
                h(cnt++) = j_[i] * j_[j];
            }
        }
        return h;
    }

    inline __HOSTDEVICE__ Vector6f ComputeJtr(float residual) {
        Vector6f jtr;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            jtr(i) = j_[i] * residual;
        }
        return jtr;
    }
};
} // cuda
} // open3d