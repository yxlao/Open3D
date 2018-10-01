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

#define __TRACE_LIFE_CYCLE__

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
	/** This is a CPU pointer for shared reference counting.
	 *  How many ImageCuda clients are using this server?
	 */
	int* ref_count_ = nullptr;

public:
	inline __DEVICE__ T& get(int x, int y);
	/** Cuda Texture is NOT helpful,
	  * especially when we want to do non-trivial interpolation
	  * for depth images.
	  * Write our own.
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
	ImageCuda();
	/** The semantic of our copy constructor (and also operator =)
	 *  is memory efficient. No moving semantic is needed.
	 */
	ImageCuda(const ImageCuda<T> &other);
	~ImageCuda();
	ImageCuda<T>& operator = (const ImageCuda<T>& other);

	int Create(int width, int height);
	void Release();
	void CopyTo(ImageCuda<T> &other) const;

	ImageCuda<T> Downsample();
	/** In-place version of downsample, defined for ImagePyramid
	 * other methods can be rewritten if needed. */
	void Downsample(ImageCuda<T> &image);

	ImageCuda<T> Gaussian(GaussianKernelOptions kernel);
	std::tuple<ImageCuda<typename T::VecTypef>,
	    ImageCuda<typename T::VecTypef>> Gradient();
	ImageCuda<typename T::VecTypef> ToFloat(float scale, float offset);

	int Upload(cv::Mat &m);
	cv::Mat Download();

	int width() const {
		return width_;
	}
	int height() const {
		return height_;
	}
	int pitch() const {
		return pitch_;
	}
	ImageCudaServer<T>& server() {
		return server_;
	}
	const ImageCudaServer<T>& server() const {
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
