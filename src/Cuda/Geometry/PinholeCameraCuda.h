//
// Created by wei on 10/3/18.
//

#pragma once

#include "VectorCuda.h"

namespace open3d {

/**
 * This class should be passed to server, but as it is a storage class, we
 * don't name it as xxxServer
 */
template<size_t N>
class PinholeCameraCuda {
public:
	int width_[N];
	int height_[N];

	float fx_[N];
	float fy_[N];
	float cx_[N];
	float cy_[N];

public:
	inline __HOSTDEVICE__ int width(size_t level = 0) {return width_[level]; }
	inline __HOSTDEVICE__ int height(size_t level = 0) {return height_[level]; }
	inline __HOSTDEVICE__ float fx(size_t level = 0) { return fx_[level]; }
	inline __HOSTDEVICE__ float fy(size_t level = 0) { return fy_[level]; }
	inline __HOSTDEVICE__ float cx(size_t level = 0) { return cx_[level]; }
	inline __HOSTDEVICE__ float cy(size_t level = 0) { return cy_[level]; }

	inline __HOSTDEVICE__ void Set(int in_width, int in_height,
								   float in_fx, float in_fy,
								   float in_cx, float in_cy) {
		for (int i = 0; i < N; ++i) {
			float factor = 1.0f / (1 << i);
			width_[i] = in_width >> i;
			height_[i] = in_height >> i;
			fx_[i] = in_fx * factor;
			fy_[i] = in_fy * factor;
			cx_[i] = in_cx * factor;
			cy_[i] = in_cy * factor;
		}
	}

	inline __HOSTDEVICE__ bool IsValid(const Vector2f &p, size_t level = 0) {
		assert(level < N);
		return p(0) >= 0 && p(0) < width_[level] - 1
			&& p(1) >= 0 && p(1) < height_[level] - 1;
	}

	inline __HOSTDEVICE__ Vector2f Projection(const Vector3f &X,
											  size_t level = 0) {
		assert(level < N);
		return Vector2f((fx_[level] * X(0)) / X(2) + cx_[level],
						(fy_[level] * X(1)) / X(2) + cy_[level]);
	}

	inline __HOSTDEVICE__ Vector3f InverseProjection(const Vector2f &p,
													 float d,
													 size_t level = 0) {
		assert(level < N);
		return Vector3f(d * (p(0) - cx_[level]) / fx_[level],
						d * (p(1) - cy_[level]) / fy_[level],
						d);
	}

	inline __HOSTDEVICE__ Vector3f InverseProjection(int x, int y, float d,
													 size_t level = 0) {
		assert(level < N);
		return Vector3f(d * (x - cx_[level]) / fx_[level],
						d * (y - cy_[level]) / fy_[level],
						d);
	}
};

typedef PinholeCameraCuda<1> MonoPinholeCameraCuda;
}
