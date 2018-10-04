//
// Created by wei on 10/3/18.
//

#ifndef OPEN3D_CAMERA_H
#define OPEN3D_CAMERA_H

#include "VectorCuda.h"

namespace three {

/**
 * This class should be passed to server, but as it is a storage class, we
 * don't name it as CameraCudaServer
 */
class CameraCuda {
public:
	int width_;
	int height_;

	float fx;
	float fy;
	float cx;
	float cy;

public:
	inline __HOSTDEVICE__ Vector2f Projection(const Vector3f &X) {
		return Vector2f((fx * X(0)) / X(2) + cx,
						(fy * X(1)) / X(2) + cy);
	}

	inline Vector3f __HOSTDEVICE__ InverseProjection(const Vector2f &p, float d) {
		return Vector3f(d * (p(0) - cx) / fx,
						d * (p(1) - cy) / fy,
						d);
	}

	inline Vector3f __HOSTDEVICE__ InverseProjection(int x, int y, float d) {
		return Vector3f(d * (x - cx) / fx,
						d * (y - cy) / fy,
						d);
	}
};

}
#endif //OPEN3D_CAMERA_H
