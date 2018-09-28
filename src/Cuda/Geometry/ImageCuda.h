//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_IMAGECUDA_H
#define OPEN3D_IMAGECUDA_H

#include "GeometryClasses.h"
#include <Cuda/Common/Common.h>

#include <cstdlib>
#include <vector_types.h>
#include <opencv2/opencv.hpp>

namespace three {

/**
 * @tparam T: uchar, uchar3, uchar4, float, float3, float4.
 * Other templates are regarded as incompatible.
 */
template<typename T>
class ImageCudaServer {
private:
	T* data_;

public:
	int width_;
	int height_;
	int pitch_;

public:
	inline __DEVICE__ T& get(int x, int y);
	/** Cuda Texture is NOT helpful,
	  * especially when we want to do non-trivial interpolation for depth images
	  */
	inline __DEVICE__ T get_interp(float x, float y);

	T* &data() {
		return data_;
	}

	friend class ImageCuda<T>;
};

enum GaussianKernelOptions {
	Gaussian3x3 = 0,
	Gaussian5x5 = 1,
	Gaussian7x7 = 2
};

template<typename T>
class ImageCuda {
private:
	ImageCudaServer<T> server_;
	int width_;
	int height_;
	int pitch_;

public:

	ImageCuda() { width_ = -1; height_=  -1; }
	~ImageCuda() = default;

	int Init(int width, int height);
	void Destroy();
	void CopyTo(ImageCuda<T> &other);

	ImageCuda<T> Downsample();
	ImageCuda<T> Gaussian(GaussianKernelOptions kernel);
	std::tuple<ImageCuda<typename T::VecTypef>,
	    ImageCuda<typename T::VecTypef>> Gradient();
	ImageCuda<typename T::VecTypef> ToFloat(float scale, float offset);

	int Upload(cv::Mat &m);
	cv::Mat Download();

	int width() {
		return width_;
	}
	int height() {
		return height_;
	}
	int pitch() {
		return pitch_;
	}
	ImageCudaServer<T>& server() {
		return server_;
	}
};

template<typename T>
__GLOBAL__
void DownsampleImageKernel(ImageCudaServer<T> src, ImageCudaServer<T> dst);

template<typename T>
__GLOBAL__
void GaussianImageKernel(ImageCudaServer<T> src, ImageCudaServer<T> dst,
	const int kernel_idx);

template<typename T>
__GLOBAL__
void ToFloatImageKernel(
	ImageCudaServer<T> src,
	ImageCudaServer<typename T::VecTypef> dst,
	float scale, float offset);

template<typename T>
__GLOBAL__
void GradientImageKernel(
	ImageCudaServer<T> src,
	ImageCudaServer<typename T::VecTypef> dx,
	ImageCudaServer<typename T::VecTypef> dy);
}

#endif //OPEN3D_IMAGECUDA_H
