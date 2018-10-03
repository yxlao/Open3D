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

//#define __TRACE_LIFE_CYCLE__

namespace three {

enum GaussianKernelSize {
	Gaussian3x3 = 0,
	Gaussian5x5 = 1,
	Gaussian7x7 = 2,
};

enum DownsampleMethod {
	BoxFilter = 0, /* Naive 2x2 */
	BoxFilterWithHoles = 1,
	GaussianFilter = 2, /* 5x5, suggested by OpenCV */
	GaussianFilterWithHoles = 3
};

/**
 * @tparam VecType: uchar, uchar3, uchar4, float, float3, float4.
 * Other templates are regarded as incompatible.
 */
template<typename VecType>
class ImageCudaServer {
private:
	VecType *data_;

public:
	int width_;
	int height_;
	int pitch_;

public:
	/** This is a CPU pointer for shared reference counting.
	 *  How many ImageCuda clients are using this server?
	 */
	int *ref_count_ = nullptr;

public:
	inline __DEVICE__ VecType &get(int x, int y);
	inline __DEVICE__ VecType &operator()(int x, int y);
	inline __DEVICE__ VecType get_interp(float x, float y);

	inline __DEVICE__
	VecType BoxFilter2x2(int x, int y);
	inline __DEVICE__
	VecType BoxFilter2x2WithHoles(int x, int y);
	inline __DEVICE__
	VecType GaussianFilter(int x, int y, int kernel_idx);
	inline __DEVICE__
	VecType GaussianFilterWithHoles(int x, int y, int kernel_idx);
	inline __DEVICE__
	VecType BilateralFilter(int x, int y, int kernel_idx, float val_sigma);
	inline __DEVICE__
	VecType BilateralFilterWithHoles(int x, int y, int kernel_idx, float val_sigma);

	/** Wish I could use std::pair here... **/
	struct Grad {
		typename VecType::VecTypef dx;
		typename VecType::VecTypef dy;
	};
	inline __DEVICE__ Grad Sobel(int x, int y);

	VecType *&data() {
		return data_;
	}

	friend class ImageCuda<VecType>;
};

template<typename VecType>
class ImageCuda {
private:
	ImageCudaServer<VecType> server_;
	int width_;
	int height_;
	int pitch_;

public:
	ImageCuda();
	/** The semantic of our copy constructor (and also operator =)
	 *  is memory efficient. No moving semantic is needed.
	 */
	ImageCuda(const ImageCuda<VecType> &other);
	~ImageCuda();
	ImageCuda<VecType> &operator=(const ImageCuda<VecType> &other);

	int Create(int width, int height);
	int Resize(int width, int height);
	void Release();
	void CopyTo(ImageCuda<VecType> &other) const;

	/** 'switch' code in kernel can be slow, manually expand it if needed. **/
	ImageCuda<VecType> Downsample(DownsampleMethod method = GaussianFilter);
	void Downsample(ImageCuda<VecType> &image,
					DownsampleMethod method = GaussianFilter);

	std::tuple<ImageCuda<typename VecType::VecTypef>,
			   ImageCuda<typename VecType::VecTypef>> Sobel();
	void Sobel(ImageCuda<typename VecType::VecTypef> &dx,
			   ImageCuda<typename VecType::VecTypef> &dy);

	ImageCuda<VecType> Gaussian(GaussianKernelSize option,
								bool with_holess = true);
	void Gaussian(ImageCuda<VecType> &image, GaussianKernelSize option,
				  bool with_holes = true);

	ImageCuda<VecType> Bilateral(GaussianKernelSize option = Gaussian5x5,
								 float val_sigma = 20.0f,
								 bool with_holes = true);
	void Bilateral(ImageCuda<VecType> &image,
				   GaussianKernelSize option = Gaussian5x5,
				   float val_sigma = 20.0f,
				   bool with_holes = true);

	ImageCuda<typename VecType::VecTypef> ToFloat(
		float scale = 1.0f, float offset = 0.0f);
	void ToFloat(ImageCuda<typename VecType::VecTypef> &image,
				 float scale, float offset);

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
	ImageCudaServer<VecType> &server() {
		return server_;
	}
	const ImageCudaServer<VecType> &server() const {
		return server_;
	}
};

template<typename VecType>
__GLOBAL__
void DownsampleImageKernel(ImageCudaServer<VecType> src,
						   ImageCudaServer<VecType> dst,
						   DownsampleMethod method);

template<typename VecType>
__GLOBAL__
void GaussianImageKernel(ImageCudaServer<VecType> src,
						 ImageCudaServer<VecType> dst,
						 const int kernel_idx,
						 bool with_holes);

template<typename VecType>
__GLOBAL__
void BilateralImageKernel(ImageCudaServer<VecType> src,
						  ImageCudaServer<VecType> dst,
						  const int kernel_idx,
						  float val_sigma,
						  bool with_holes);

template<typename VecType>
__GLOBAL__
void SobelImageKernel(ImageCudaServer<VecType> src,
					  ImageCudaServer<typename VecType::VecTypef> dx,
					  ImageCudaServer<typename VecType::VecTypef> dy);

template<typename VecType>
__GLOBAL__
void ToFloatImageKernel(ImageCudaServer<VecType> src,
						ImageCudaServer<typename VecType::VecTypef> dst,
						float scale, float offset);
}

#endif //OPEN3D_IMAGECUDA_H
