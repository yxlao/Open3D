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
}

#endif //OPEN3D_IMAGECUDA_H
