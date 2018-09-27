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
 * Eigen is (quite) incompatible with CUDA. Write my own version to
 * do this.
 */
class Vector3i {
private:
	union {
		int3 vec;
		int arr[3];
	} val_;

public:
	__HOSTDEVICE__ Vector3i() {
		val_.vec = make_int3(0);
	}
	__HOSTDEVICE__ Vector3i(int val) {
		val_.vec = make_int3(val);
	}
	__HOSTDEVICE__ Vector3i(int v1, int v2, int v3) {
		val_.vec = make_int3(v1, v2, v3);
	}
	__HOSTDEVICE__ int& operator() (int i) {
		assert(i >= 0 && i < 3);
		return val_.arr[i];
	}
	__HOSTDEVICE__ const int &operator() (int i) const {
		assert(i >= 0 && i < 3);
		return val_.arr[i];
	}
	__HOSTDEVICE__ bool operator == (const Vector3i &other) const {
		return (*this)(0) == other(0)
		&& (*this)(1) == other(1)
		&& (*this)(2) == other(2);
	}
	__HOST__ Eigen::Vector3i ToEigen() {
		Eigen::Vector3i vec;
		vec << (*this)(0), (*this)(1), (*this)(2);
		return vec;
	}
	__HOST__ void FromEigen(Eigen::Vector3i& other) {
		val_.arr[0] = other(0);
		val_.arr[1] = other(1);
		val_.arr[2] = other(2);
	}
};

}
#endif //OPEN3D_VECTOR_H
