//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_VECTOR_H
#define OPEN3D_VECTOR_H

#include <Cuda/Common/Common.h>
#include <Cuda/Common/HelperMath.h>
#include <Eigen/Eigen>

#include <cassert>

namespace three {

/**
 * Eigen is (quite) incompatible with CUDA.
 * Built-in data structures (int3, float3, ...) does not support generic
 * programming.
 * Write my own version to do this.
 */

template<typename T, size_t N>
class Vector {
public:
	T v[N];

public:
	typedef T ValType;
	typedef Vector<T, N> VecType;
	typedef Vector<float, N> VecTypef;

	/** Conversions **/
	__HOSTDEVICE__ static VecTypef Vectorf() {
		return VecTypef();
	}
	__HOSTDEVICE__ VecTypef ToVectorf() {
		VecTypef ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = (float) v[i];
		}
		return ret;
	}
	__HOSTDEVICE__ void FromVectorf(const VecTypef &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] = T(other.v[i]);
		}
	}

	__HOSTDEVICE__ Vector() {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] = T(0);
		}
	}

	/**
	 * WARNING! This Createializer is special !!!
	 * @param v0
	 */
	__HOSTDEVICE__ Vector(T v0) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] = v0;
		}
	}
	__HOSTDEVICE__ Vector(T v0, T v1) {
		assert(N > 1);
		v[0] = v0, v[1] = v1;
	}
	__HOSTDEVICE__ Vector(T v0, T v1, T v2) {
		assert(N > 2);
		v[0] = v0, v[1] = v1, v[2] = v2;
	}
	__HOSTDEVICE__ Vector(T v0, T v1, T v2, T v3) {
		assert(N > 3);
		v[0] = v0, v[1] = v1, v[2] = v2, v[3] = v3;
	}
	__HOSTDEVICE__ inline T& operator[] (size_t i) {
		assert(i < N);
		return v[i];
	}
	__HOSTDEVICE__ inline const T &operator[] (size_t i) const {
		assert(i < N);
		return v[i];
	}
	__HOSTDEVICE__ bool operator == (const VecType &other) const {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (size_t i = 0; i < N; ++i) {
			if (v[i] != other.v[i]) return false;
		}
		return true;
	}
	__HOSTDEVICE__ VecType operator + (const VecType &other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] + other.v[i];
		}
		return ret;
	}
	__HOSTDEVICE__ void operator += (const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] += other.v[i];
		}
	}
	__HOSTDEVICE__ VecType operator - (const VecType &other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] - other.v[i];
		}
		return ret;
	}
	__HOSTDEVICE__ void operator -= (const VecType &other)  {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] -= other.v[i];
		}
	}

	__HOSTDEVICE__ VecType operator * (const VecType &other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] * other.v[i];
		}
		return ret;
	}

	__HOSTDEVICE__ VecType operator * (const float other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] * other;
		}
		return ret;
	}

	__HOSTDEVICE__ void operator *= (const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] *= other.v[i];
		}
	}

	__HOSTDEVICE__ void operator *= (const float other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] *= other;
		}
	}

	__HOSTDEVICE__ VecType operator / (const VecType &other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] / other.v[i];
		}
		return ret;
	}

	__HOSTDEVICE__ VecType operator / (const float other) const {
		VecType ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i] / other;
		}
		return ret;
	}

	__HOSTDEVICE__ void operator /= (const VecType &other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] /= other.v[i];
		}
	}

	__HOSTDEVICE__ void operator /= (const float other) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			v[i] /= other;
		}
	}

	__HOSTDEVICE__ Vector<T, N+1> homogeneous() {
		Vector<T, N+1> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N; ++i) {
			ret.v[i] = v[i];
		}
		ret.v[N] = T(1);
		return ret;
	}

	__HOSTDEVICE__ Vector<T, N-1> hnormalized() {
		assert(typeid(T) == typeid(float) && N > 1);
		Vector<T, N-1> ret;
#ifdef __CUDACC__
#pragma unroll 1
#endif
		for (int i = 0; i < N - 1; ++i) {
			ret.v[i] = v[i] / v[N-1];
		}
		return ret;
	}
};

typedef Vector<int, 3> Vector3i;

typedef Vector<short, 1> Vector1s;
typedef Vector<uchar, 1> Vector1b;
typedef Vector<uchar, 3> Vector3b;
typedef Vector<uchar, 4> Vector4b;
typedef Vector<float, 1> Vector1f;
typedef Vector<float, 3> Vector3f;
typedef Vector<float, 4> Vector4f;

}
#endif //OPEN3D_VECTOR_H
