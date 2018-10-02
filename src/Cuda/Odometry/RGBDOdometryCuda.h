//
// Created by wei on 10/1/18.
//

#ifndef OPEN3D_RGBDODOMETRY_H
#define OPEN3D_RGBDODOMETRY_H

#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Eigen/Eigen>

namespace three {

template<typename T, size_t N>
class RGBDOdometryCudaServer {
private:
	ImagePyramidCudaServer<T, N> source_;
	ImagePyramidCudaServer<T, N> target_;

public:

};

template<typename T, size_t N>
class RGBDOdometryCuda {
private:
	RGBDOdometryCudaServer server_;

public:
	void Create();
	void Release();

	void Apply(cv::Mat &source, cv::Mat &target);
	void Apply(ImageCuda<T>& source, ImageCuda<T>& target);

	RGBDOdometryCudaServer& server() {
		return server_;
	}
	const RGBDOdometryCudaServer& server() const {
		return server_;
	}
};
}
#endif //OPEN3D_RGBDODOMETRY_H
