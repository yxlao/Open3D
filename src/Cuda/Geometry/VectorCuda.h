//
// Created by wei on 9/27/18.
//

#pragma once

#include <Cuda/Common/Common.h>
#include <Cuda/Common/HelperMath.h>
#include <Eigen/Eigen>

#include <cassert>

namespace open3d {

/**
 * Eigen is (quite) incompatible with CUDA -- countless warnings.
 * Built-in data structures (int3, float3, ...) does not support generic
 * programming.
 * Write my own version to do this.
 */

template<typename T, size_t N>
class VectorCuda {
public:
    T v[N];

public:
    typedef T ValType;
    typedef VectorCuda<T, N> VecType;
    typedef VectorCuda<float, N> VecTypef;

    /** Conversions **/
    inline __HOSTDEVICE__ static VecTypef Vectorf() {
        return VecTypef();
    }
    inline __HOSTDEVICE__ VecTypef ToVectorf() {
        VecTypef ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = (float) v[i];
        }
        return ret;
    }
    inline __HOSTDEVICE__ void FromVectorf(const VecTypef &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = T(other.v[i]);
        }
    }

    /** Constants **/
    inline __HOSTDEVICE__ VectorCuda<T, N> static Zeros() {
        return VectorCuda<T, N>(0);
    }
    inline __HOSTDEVICE__ VectorCuda<T, N> static Ones() {
        return VectorCuda<T, N>(1);
    }

    /** Constructors **/
    inline __HOSTDEVICE__ VectorCuda(const VecTypef &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = T(other.v[i]);
        }
    }

    inline __HOSTDEVICE__ VectorCuda() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = T(0);
        }
    }

    /**
     * WARNING! This initializer is special !!!
     * @param v0
     */
    inline __HOSTDEVICE__ explicit VectorCuda(T v0) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = v0;
        }
    }
    inline __HOSTDEVICE__ VectorCuda(T v0, T v1) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 1);
#endif
        v[0] = v0, v[1] = v1;
    }
    inline __HOSTDEVICE__ VectorCuda(T v0, T v1, T v2) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 2);
#endif
        v[0] = v0, v[1] = v1, v[2] = v2;
    }
    inline __HOSTDEVICE__ VectorCuda(T v0, T v1, T v2, T v3) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 3);
#endif
        v[0] = v0, v[1] = v1, v[2] = v2, v[3] = v3;
    }
    inline __HOSTDEVICE__ T &operator()(size_t i) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return v[i];
    }
    inline __HOSTDEVICE__ const T &operator()(size_t i) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return v[i];
    }
    inline __HOSTDEVICE__ bool operator==(const VecType &other) const {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < N; ++i) {
            if (v[i] != other.v[i]) return false;
        }
        return true;
    }
    inline __HOSTDEVICE__ bool operator!=(const VecType &other) const {
        return !((*this) == other);
    }
    inline __HOSTDEVICE__ VecType operator+(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] + other.v[i];
        }
        return ret;
    }
    inline __HOSTDEVICE__ void operator+=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] += other.v[i];
        }
    }

    inline __HOSTDEVICE__ VecType operator-() const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = -v[i];
        }
        return ret;
    }

    inline __HOSTDEVICE__ VecType operator-(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] - other.v[i];
        }
        return ret;
    }
    inline __HOSTDEVICE__ void operator-=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] -= other.v[i];
        }
    }

    inline __HOSTDEVICE__ VecType operator*(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] * other.v[i];
        }
        return ret;
    }

    inline __HOSTDEVICE__ VecType operator*(const float other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] * other;
        }
        return ret;
    }

    inline __HOSTDEVICE__ void operator*=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] *= other.v[i];
        }
    }

    inline __HOSTDEVICE__ void operator*=(const float other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] *= other;
        }
    }

    inline __HOSTDEVICE__ VecType operator/(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] / other.v[i];
        }
        return ret;
    }

    inline __HOSTDEVICE__ VecType operator/(const float other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] / other;
        }
        return ret;
    }

    inline __HOSTDEVICE__ void operator/=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] /= other.v[i];
        }
    }

    inline __HOSTDEVICE__ void operator/=(const float other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] /= other;
        }
    }

    inline __HOSTDEVICE__ VectorCuda<T, N + 1> homogeneous() {
        VectorCuda<T, N + 1> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i];
        }
        ret.v[N] = T(1);
        return ret;
    }

    inline __HOSTDEVICE__ VectorCuda<T, N - 1> hnormalized() {
        VectorCuda<T, N - 1> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N - 1; ++i) {
            ret.v[i] = v[i] / v[N - 1];
        }
        return ret;
    }

    inline __HOSTDEVICE__ float dot(const VecType &other) {
        float sum = 0;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            sum += float(v[i]) * float(other.v[i]);
        }
        return sum;
    }

    inline __HOSTDEVICE__ float norm() {
        return sqrtf(dot(*this));
    }

    inline __HOSTDEVICE__ VectorCuda<T, N> normalized() {
        float n = norm();
        VectorCuda<T, N> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] / n;
        }
        return ret;
    }

    /** CPU CODE **/
    inline void FromEigen(Eigen::Matrix<T, N, 1> &other) {
        for (int i = 0; i < N; ++i) {
            v[i] = other(i);
        }
    }

    inline Eigen::Matrix<T, N, 1> ToEigen() {
        Eigen::Matrix<T, N, 1> ret;
        for (int i = 0; i < N; ++i) {
            ret(i) = v[i];
        }
        return ret;
    }
};

template<typename T, size_t N>
inline __HOSTDEVICE__
VectorCuda<T, N> operator*(float s, const VectorCuda<T, N> &vec) {
    return vec * s;
}

typedef VectorCuda<int, 2> Vector2i;
typedef VectorCuda<int, 3> Vector3i;
typedef VectorCuda<int, 4> Vector4i;

typedef VectorCuda<short, 1> Vector1s;

typedef VectorCuda<uchar, 1> Vector1b;
typedef VectorCuda<uchar, 3> Vector3b;
typedef VectorCuda<uchar, 4> Vector4b;

typedef VectorCuda<float, 1> Vector1f;
typedef VectorCuda<float, 2> Vector2f;
typedef VectorCuda<float, 3> Vector3f;
typedef VectorCuda<float, 4> Vector4f;
typedef VectorCuda<float, 6> Vector6f;

}
