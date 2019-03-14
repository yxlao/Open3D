//
// Created by wei on 9/27/18.
//

#pragma once

#include <src/Cuda/Common/Common.h>
#include <Eigen/Eigen>

#include <cassert>

namespace open3d {
namespace cuda {
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
    typedef VectorCuda<int, N> VecTypei;
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
    __HOSTDEVICE__ inline VectorCuda<T, N + 1> homogeneous() const {
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

    __HOSTDEVICE__ inline VectorCuda<T, N - 1> hnormalized() const {
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

    __HOSTDEVICE__ inline float dot(const VecType &other) const {
        float sum = 0;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < N; ++i) {
            sum += float(v[i]) * float(other.v[i]);
        }
        return sum;
    }

    __HOSTDEVICE__ inline VecType cross(const VecType &other) const {
        static_assert(N == 3, "Invalid vector dimension");

        return VecType(
            v[1] * other.v[2] - v[2] * other.v[1],
            v[2] * other.v[0] - v[0] * other.v[2],
            v[0] * other.v[1] - v[1] * other.v[0]);
    }

    __HOSTDEVICE__ inline float norm() const {
        return sqrtf(dot(*this));
    }

    __HOSTDEVICE__ inline VectorCuda<T, N> normalized() const {
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

template<typename T, size_t M, size_t N>
class LDLT;

template<typename T, size_t M, size_t N>
class MatrixCuda {
public:
    T v[M * N];

public:
    __HOSTDEVICE__ MatrixCuda() {}

    __HOSTDEVICE__ explicit MatrixCuda(const T &val) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < M * N; ++i) {
            v[i] = val;
        }
    }

    __HOSTDEVICE__ T &at(size_t i, size_t j) {
        return v[i * N + j];
    }

    __HOSTDEVICE__ const T& at(size_t i, size_t j) const {
        return v[i * N + j];
    }

    __HOSTDEVICE__ T &operator()(size_t i, size_t j) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < M && j < N);
#endif
        return v[i * N + j];
    }

    __HOSTDEVICE__ const T& operator() (size_t i, size_t j) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < M && j < N);
#endif
        return v[i * N + j];
    }

    __HOSTDEVICE__ MatrixCuda<T, M, N> &operator=(
        const MatrixCuda<T, M, N> &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < M * N; ++i) {
            v[i] = other.v[i];
        }

        return *this;
    }

    __HOSTDEVICE__ VectorCuda<T, M> &operator*(const VectorCuda<T, N> &vec) {
        VectorCuda<T, M> res(0);
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                res(i) += at(i, j) * vec(j);
            }
        }
        return res;
    }

    __HOSTDEVICE__ LDLT<T, M, N> ldlt() {
        return LDLT<T, M, N>(*this);
    }

    /* CPU code */
    void FromEigen(const Eigen::Matrix<double, M, N> &other) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                at(i, j) = T(other(i, j));
            }
        }
    }

    Eigen::Matrix<double, M, N> ToEigen() {
        Eigen::Matrix<double, M, N> ret;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ret(i, j) = double(at(i, j));
            }
        }
        return ret;
    }
};

template<typename T, size_t M, size_t N>
class LDLT {
public:
    bool valid = true;
    MatrixCuda<T, N, N> L;
    VectorCuda<T, N> D;

    /* https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition */
    /* http://mathforcollege.com/nm/mws/gen/04sle/mws_gen_sle_txt_cholesky.pdf */
    /* A = LDL^T */
    __HOSTDEVICE__ LDLT(const MatrixCuda<T, M, N> &A) {
        static_assert(M == N, "M != N");

        for (int i = 0; i < N; ++i) {
            /* Update L */
            for (int j = 0; j < i; ++j) {
                float s = 0;
                for (int k = 0; k < j; ++k) {
                    s += L(i, k) * L(j, k) * D(k);
                }
                L(i, j) = 1.0f / D(j) * (A(i, j) - s);
            }

            /* Initialize rest of the row */
            L(i, i) = 1.0f;
            for (int j = i + 1; j < N; ++j) {
                L(i, j) = 0;
            }

            /* Update D */
            float s = 0;
            for (int k = 0; k < i; ++k) {
                s += L(i, k) * L(i, k) * D(k);
            }
            D(i) = A(i, i) - s;

            /* Singular condition */
            if (fabs(D(i)) < 1e-4f) {
                valid = false;
                return;
            }
        }
    }

    /* LDL^T = b */
    __HOSTDEVICE__ VectorCuda<float, N> Solve(const VectorCuda<float, N> &b) {
        /* Solve Ly = b */
        VectorCuda<float, N> y, x;

        if (!valid) return VectorCuda<float, N>(0);

        for (int i = 0; i < N; ++i) {
            y(i) = b(i);
            for (int j = 0; j < i; ++j) {
                y(i) -= L(i, j) * y(j);
            }
        }

        /* Solve D L^T x = y */
        for (int i = int(N - 1); i >= 0; --i) {
            x(i) = y(i) / D(i);
            for (int j = i + 1; j < N; ++j) {
                x(i) -= L(j, i) * x(j);
            }
        }

        return x;
    }
};

typedef MatrixCuda<float, 3, 3> Matrix3f;
typedef LDLT<float, 3, 3> LDLT3;

} // cuda
} // open3d