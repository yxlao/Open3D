//
// Created by wei on 10/3/18.
//

#pragma once

#include "LinearAlgebraCuda.h"

#include <Eigen/Eigen>

namespace open3d {

namespace cuda {
class TransformCuda {
private:
    float m_[3][4];

public:
    __HOSTDEVICE__ TransformCuda() {};
    __HOSTDEVICE__ static TransformCuda Identity() {
        TransformCuda ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < 3; ++i)
#ifdef __CUDACC__
#pragma unroll 1
#endif
            for (size_t j = 0; j < 4; ++j)
                ret(i, j) = (i == j) ? 1 : 0;
        return ret;
    }

    __HOSTDEVICE__ inline float &operator()(size_t i, size_t j) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < 3 && j < 4);
#endif
        return m_[i][j];
    }
    __HOSTDEVICE__ inline const float &operator()(size_t i, size_t j) const {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(i < 3 && j < 4);
#endif
        return m_[i][j];
    }

    __HOSTDEVICE__ inline Vector3f operator*(const Vector3f &v) {
        return Vector3f(
            m_[0][0] * v(0) + m_[0][1] * v(1) + m_[0][2] * v(2) + m_[0][3],
            m_[1][0] * v(0) + m_[1][1] * v(1) + m_[1][2] * v(2) + m_[1][3],
            m_[2][0] * v(0) + m_[2][1] * v(1) + m_[2][2] * v(2) + m_[2][3]);
    }

    __HOSTDEVICE__ TransformCuda operator*(const TransformCuda &other) {
        TransformCuda ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < 3; ++i) {
            ret(i, 0) = m_[i][0] * other(0, 0) + m_[i][1] * other(1, 0) + m_[i][2] * other(2, 0);
            ret(i, 1) = m_[i][0] * other(0, 1) + m_[i][1] * other(1, 1) + m_[i][2] * other(2, 1);
            ret(i, 2) = m_[i][0] * other(0, 2) + m_[i][1] * other(1, 2) + m_[i][2] * other(2, 2);
            ret(i, 3) = m_[i][0] * other(0, 3) + m_[i][1] * other(1, 3) + m_[i][2] * other(2, 3) + m_[i][3];
        }
        return ret;
    }

    __HOSTDEVICE__ TransformCuda &operator=(const TransformCuda &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < 3; ++i) {
            m_[i][0] = other(i, 0);
            m_[i][1] = other(i, 1);
            m_[i][2] = other(i, 2);
            m_[i][3] = other(i, 3);
        }
        return *this;
    }

    __HOSTDEVICE__ TransformCuda Inverse() {
        TransformCuda ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
        for (size_t i = 0; i < 3; ++i) {
            ret(i, 0) = m_[0][i];
            ret(i, 1) = m_[1][i];
            ret(i, 2) = m_[2][i];
            ret(i, 3) = -(m_[0][i] * m_[0][3] + m_[1][i] * m_[1][3] + m_[2][i] * m_[2][3]);
        }
        return ret;
    }

    __HOSTDEVICE__ inline Vector3f Rotate(const Vector3f &v) {
        return Vector3f(
            m_[0][0] * v(0) + m_[0][1] * v(1) + m_[0][2] * v(2),
            m_[1][0] * v(0) + m_[1][1] * v(1) + m_[1][2] * v(2),
            m_[2][0] * v(0) + m_[2][1] * v(1) + m_[2][2] * v(2));
    }

    inline __HOSTDEVICE__ void SetTranslation(const Vector3f &translation) {
        m_[0][3] = translation(0);
        m_[1][3] = translation(1);
        m_[2][3] = translation(2);
    }

    /** CPU ONLY **/
    void FromEigen(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m_[i][j] = float(R(i, j));
            }
            m_[i][3] = float(t(i));
        }
    }

    void FromEigen(const Eigen::Matrix4d &T) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                m_[i][j] = float(T(i, j));
            }
        }
    }

    Eigen::Matrix4d ToEigen() {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                T(i, j) = double(m_[i][j]);
            }
        }
        return T;
    }
};
} // cuda
} // open3d