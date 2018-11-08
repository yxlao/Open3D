//
// Created by wei on 9/27/18.
//

#pragma once

#include <Cuda/Common/Common.h>
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
    typedef VectorCuda<int, N>   VecTypei;
    typedef VectorCuda<ushort, N> VecTypes;
    typedef VectorCuda<uchar, N> VecTypeb;

    /*********************** Conversions ***********************/
    __HOSTDEVICE__ inline static VecTypef Vectorf() {
        return VecTypef();
    }
    __HOSTDEVICE__ inline static VecTypei Vectori() {
        return VecTypei();
    }
    __HOSTDEVICE__ inline static VecTypes Vectors() {
        return VecTypes();
    }
    __HOSTDEVICE__ inline static VecTypeb Vectorb() {
        return VecTypeb();
    }

    __HOSTDEVICE__ inline VecTypef ToVectorf() const {
        VecTypef ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = float(v[i]);
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecTypei ToVectori() const {
        VecTypei ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = int(v[i]);
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecTypes ToVectors() const {
        VecTypes ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = ushort(fminf(v[i], 65535));
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecTypeb ToVectorb() const {
        VecTypeb ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = uchar(fminf(v[i], 255));
        }
        return ret;
    }

    __HOSTDEVICE__ inline static VectorCuda<T, N> FromVectorf(
        const VecTypef &other) {
        VectorCuda<T, N> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret(i) = T(other.v[i]);
        }
        return ret;
    }

    /*********************** Constants ***********************/
    __HOSTDEVICE__ inline VectorCuda<T, N> static Zeros() {
        return VectorCuda<T, N>(0);
    }
    __HOSTDEVICE__ inline VectorCuda<T, N> static Ones() {
        return VectorCuda<T, N>(1);
    }
    __HOSTDEVICE__ inline bool IsZero() {
        bool is_zero = true;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            is_zero = is_zero && v[i] == 0;
        }
        return is_zero;
    }


    /*********************** Constructors ***********************/
    __HOSTDEVICE__ inline VectorCuda() {};
    __HOSTDEVICE__ inline VectorCuda(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = other.v[i];
        }
    }

    /**
     * WARNING! This initializer is special !!!
     * @param v0
     */
    __HOSTDEVICE__ inline explicit VectorCuda(T v0) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = v0;
        }
    }
    __HOSTDEVICE__ inline VectorCuda(T v0, T v1) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 1);
#endif
        v[0] = v0, v[1] = v1;
    }
    __HOSTDEVICE__ inline VectorCuda(T v0, T v1, T v2) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 2);
#endif
        v[0] = v0, v[1] = v1, v[2] = v2;
    }
    __HOSTDEVICE__ inline VectorCuda(T v0, T v1, T v2, T v3) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 3);
#endif
        v[0] = v0, v[1] = v1, v[2] = v2, v[3] = v3;
    }

    /** Access **/
    __HOSTDEVICE__ inline T &operator()(size_t i) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return v[i];
    }
    __HOSTDEVICE__ inline const T &operator()(size_t i) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < N);
#endif
        return v[i];
    }

    /** Comparison **/
    __HOSTDEVICE__ inline bool operator==(const VecType &other) const {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < N; ++i) {
            if (v[i] != other.v[i]) return false;
        }
        return true;
    }
    __HOSTDEVICE__ inline bool operator!=(const VecType &other) const {
        return !((*this) == other);
    }

    /** Arithmetic operators **/
    __HOSTDEVICE__ inline VecType operator+(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] + other.v[i];
        }
        return ret;
    }
    __HOSTDEVICE__ inline void operator+=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] += other.v[i];
        }
    }

    __HOSTDEVICE__ inline VecType operator-() const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = -v[i];
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecType operator-(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] - other.v[i];
        }
        return ret;
    }
    __HOSTDEVICE__ inline void operator-=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] -= other.v[i];
        }
    }

    __HOSTDEVICE__ inline VecType operator*(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] * other.v[i];
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecType operator*(const T val) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = T(v[i] * val);
        }
        return ret;
    }

    __HOSTDEVICE__ inline void operator*=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] *= other.v[i];
        }
    }

    __HOSTDEVICE__ inline void operator*=(const T val) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = T(v[i] * val);
        }
    }

    __HOSTDEVICE__ inline VecType operator/(const VecType &other) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = v[i] / other.v[i];
        }
        return ret;
    }

    __HOSTDEVICE__ inline VecType operator/(const T val) const {
        VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            ret.v[i] = T(v[i] / val);
        }
        return ret;
    }

    __HOSTDEVICE__ inline void operator/=(const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] /= other.v[i];
        }
    }

    __HOSTDEVICE__ inline void operator/=(const T val) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            v[i] = T(v[i] / val);
        }
    }

    /** Linear algebraic operations **/
    __HOSTDEVICE__ inline VectorCuda<T, N + 1> homogeneous() {
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

    __HOSTDEVICE__ inline VectorCuda<T, N - 1> hnormalized() {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(N > 1);
#endif

        VectorCuda<T, N - 1> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N - 1; ++i) {
            ret.v[i] = v[i] / v[N - 1];
        }
        return ret;
    }

    __HOSTDEVICE__ inline float dot(const VecType &other) {
        float sum = 0;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            sum += float(v[i]) * float(other.v[i]);
        }
        return sum;
    }

    __HOSTDEVICE__ inline float norm() {
        return sqrtf(dot(*this));
    }

    __HOSTDEVICE__ inline VectorCuda<T, N> normalized() {
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
    inline void FromEigen(Eigen::Matrix<double, N, 1> &other) {
        for (int i = 0; i < N; ++i) {
            v[i] = T(other(i));
        }
    }

    inline Eigen::Matrix<double, N, 1> ToEigen() {
        Eigen::Matrix<double, N, 1> ret;
        for (int i = 0; i < N; ++i) {
            ret(i) = double(v[i]);
        }
        return ret;
    }
};

template<typename T, size_t N>
__HOSTDEVICE__ inline
VectorCuda<T, N> operator*(T s, const VectorCuda<T, N> &vec) {
    return vec * s;
}

typedef VectorCuda<int, 2> Vector2i;
typedef VectorCuda<int, 3> Vector3i;
typedef VectorCuda<int, 4> Vector4i;

typedef VectorCuda<ushort, 1> Vector1s;

typedef VectorCuda<uchar, 1> Vector1b;
typedef VectorCuda<uchar, 3> Vector3b;
typedef VectorCuda<uchar, 4> Vector4b;

typedef VectorCuda<float, 1> Vector1f;
typedef VectorCuda<float, 2> Vector2f;
typedef VectorCuda<float, 3> Vector3f;
typedef VectorCuda<float, 4> Vector4f;
typedef VectorCuda<float, 6> Vector6f;

}
