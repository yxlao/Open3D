//
// Created by wei on 10/3/18.
//

#ifndef OPEN3D_TRANSFORMCUDA_H
#define OPEN3D_TRANSFORMCUDA_H

#include "VectorCuda.h"
#include <Eigen/Eigen>

namespace three {

class TransformCuda {
private:
	float m_[3][4];

public:
	typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f;

	inline __HOSTDEVICE__ float &operator() (size_t i, size_t j) {
		assert(i < 3 && j < 4);
		return m_[i][j];
	}
	inline __HOSTDEVICE__ const float &operator() (size_t i, size_t j) const {
		assert(i < 3 && j < 4);
		return m_[i][j];
	}


	inline __HOSTDEVICE__ Vector3f operator*(const Vector3f &v) {
		Vector3f ret(0);
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (size_t i = 0; i < 3; ++i) {
			ret(i) = m_[i][0] * v(0) + m_[i][1] * v(1) + m_[i][2] * v(2)
				+ m_[i][3];
		}
		return ret;
	}

	inline __HOSTDEVICE__ TransformCuda& operator=(const TransformCuda &other) {
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

	inline TransformCuda __HOSTDEVICE__ inv() {
		TransformCuda ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (size_t i = 0; i < 3; ++i) {
			ret(i, 0) = m_[0][i];
			ret(i, 1) = m_[1][i];
			ret(i, 2) = m_[2][i];
			ret(i, 3) = -(m_[0][i] * m_[0][3]
				+ m_[1][i] * m_[1][3]
				+ m_[2][i] * m_[2][3]);
		}
		return ret;
	}

	/** CPU ONLY **/
	inline void FromEigen(Eigen::Matrix3f &R, Eigen::Vector3f &t) {
		for (size_t i = 0; i < 3; ++i) {
			for (size_t j = 0; j < 3; ++j) {
				m_[i][j] = R(i, j);
			}
			m_[i][3] = t(i);
		}
	}

	inline void FromEigen(Matrix4f &T) {
		for (size_t i = 0; i < 3; ++i) {
			for (size_t j = 0; j < 4; ++j) {
				m_[i][j] = T(i, j);
			}
		}
	}

	inline Matrix4f ToEigen() {
		Matrix4f T = Matrix4f::Identity();
		for (size_t i = 0; i < 3; ++i) {
			for (size_t j = 0; j < 4; ++j) {
				T(i, j) = m_[i][j];
			}
		}
		return T;
	}
};
}

#endif //OPEN3D_TRANSFORMCUDA_H
