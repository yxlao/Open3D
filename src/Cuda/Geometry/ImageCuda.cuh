/**
 * Created by wei on 18-4-9.
 */

#ifndef _IMAGE_CUDA_CUH_
#define _IMAGE_CUDA_CUH_

#include "ImageCuda.h"
#include "Vector.h"
#include <iostream>
#include <driver_types.h>
#include <Cuda/Common/UtilsCuda.h>
#include <vector_types.h>
#include <Core/Core.h>

namespace three {

/**
 * Server end
 */
template<typename T>
__device__
T &ImageCudaServer<T>::get(int x, int y) {
	assert(x >= 0 && x < width_);
	assert(y >= 0 && y < height_);
	T *value = (T *) ((char *) data_ + y * pitch_) + x;
	return (*value);
}

/**
 * Naive interpolation without considering holes in depth images.
 */
template<class T>
__device__
T ImageCudaServer<T>::get_interp(float x, float y) {
	assert(x >= 0 && x <= width_ - 2);
	assert(y >= 0 && y <= height_ - 2);

	int x0 = (int)floor(x), y0 = (int)floor(y);
	float a = x - x0, b = y - y0;
	return (1 - a) * (1 - b) * get(x0, y0)
	     + (1 - a) * b * get(x0, y0 + 1)
	     + a * b * get(x0 + 1, y0 + 1)
	     + a * (1 - b) * get(x0 + 1, y0);
}

/**
 * Client end
 */
template<typename T>
int ImageCuda<T>::Init(int width, int height) {
	assert(width > 0 && height > 0);

	width_ = width;
	height_ = height;
	pitch_ = 0;

	size_t pitch_size_t = 0;
	CheckCuda(cudaMallocPitch((void**)(&server_.data_), &pitch_size_t,
		sizeof(T) * width_, height_));

	pitch_ = (int) pitch_size_t;
	server_.width_ = width_;
	server_.height_ = height_;
	server_.pitch_ = pitch_;

	return 0;
}

template<class T>
void ImageCuda<T>::Destroy() {
	if (width_ > 0 && height_ > 0) {
		CheckCuda(cudaFree(server_.data_));
	}
	width_ = -1;
	height_ = -1;
}

template<class T>
void ImageCuda<T>::CopyTo(ImageCuda<T> &other) {
	if (other.width() != width_ || other.height() != height_) {
		if (other.width() > 0 && other.height() > 0) {
			other.Destroy();
		}
		other.Init(width_, height_);
	}

	CheckCuda(cudaMemcpy2D(other.server().data(), other.pitch(),
		server_.data_, pitch_, sizeof(T) * width_, height_,
		cudaMemcpyDeviceToDevice));
}

template<class T>
int ImageCuda<T>::Upload(cv::Mat &m) {
	assert(m.rows > 0 && m.cols > 0);

	/* Type checking */
	if (typeid(T) == typeid(Vector1s)) {
		assert(m.type() == CV_16UC1);
	} else if (typeid(T) == typeid(Vector4b)) {
		assert(m.type() == CV_8UC4);
	} else if (typeid(T) == typeid(Vector3b)) {
		assert(m.type() == CV_8UC3);
	} else if (typeid(T) == typeid(Vector1b)) {
		assert(m.type() == CV_8UC1);
	} else if (typeid(T) == typeid(Vector4f)) {
		assert(m.type() == CV_32FC4);
	} else if (typeid(T) == typeid(Vector3f)) {
		assert(m.type() == CV_32FC3);
	} else if (typeid(T) == typeid(Vector1f)) {
		assert(m.type() == CV_32FC1);
	} else {
		PrintError("Unsupported format %d!\n");
		return -1;
	}

	/* CUDA memory reuse -- don't re-allocate! */
	if (width_ != m.cols || height_ != m.rows) {
		if (width_ > 0 && height_ > 0) {
			Destroy();
		}

		Init(m.cols, m.rows);
	}

	CheckCuda(cudaMemcpy2D(server_.data_, pitch_, m.data, m.step,
		sizeof(T) * m.cols, m.rows,
		cudaMemcpyHostToDevice));
	return 0;
}

template<class T>
cv::Mat ImageCuda<T>::Download() {
	cv::Mat m;
	if (typeid(T) == typeid(Vector1s)) {
		m = cv::Mat(height_, width_, CV_16UC1);
	} else if (typeid(T) == typeid(Vector4b)) {
		m = cv::Mat(height_, width_, CV_8UC4);
	} else if (typeid(T) == typeid(Vector3b)) {
		m = cv::Mat(height_, width_, CV_8UC3);
	} else if (typeid(T) == typeid(Vector1b)) {
		m = cv::Mat(height_, width_, CV_8UC1);
	} else if (typeid(T) == typeid(Vector4f)) {
		m = cv::Mat(height_, width_, CV_32FC4);
	} else if (typeid(T) == typeid(Vector3f)) {
		m = cv::Mat(height_, width_, CV_32FC3);
	} else if (typeid(T) == typeid(Vector1f)) {
		m = cv::Mat(height_, width_, CV_32FC1);
	} else {
		PrintError("Unsupported format %d!\n");
		return m;
	}

	CheckCuda(cudaMemcpy2D(m.data, m.step, server_.data_, pitch_,
		sizeof(T) * width_, height_,
		cudaMemcpyDeviceToHost));
	return m;
}

template<typename T>
ImageCuda<T> ImageCuda<T>::Downsample() {
	ImageCuda<T> dst;
	dst.Init(width_ / 2, height_ / 2);

	const dim3 blocks(
		UPPER_ALIGN(dst.width(), THREAD_2D_UNIT),
		UPPER_ALIGN(dst.height(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	DownsampleImageKernel<<<blocks, threads>>>(server_, dst.server());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());

	return dst;
}

template<typename T>
ImageCuda<T> ImageCuda<T>::Gaussian(GaussianKernelOptions kernel) {
	ImageCuda<T> dst;
	dst.Init(width_, height_);

	const dim3 blocks(
		UPPER_ALIGN(dst.width(), THREAD_2D_UNIT),
		UPPER_ALIGN(dst.height(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	GaussianImageKernel<<<blocks, threads>>>(
		server_, dst.server(), (int)kernel);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());

	return dst;
}

template<typename T>
ImageCuda<typename T::VecTypef> ImageCuda<T>::ToFloat(
	float scale, float offset) {
	ImageCuda<typename T::VecTypef> dst;
	dst.Init(width_, height_);

	const dim3 blocks(
		UPPER_ALIGN(dst.width(), THREAD_2D_UNIT),
		UPPER_ALIGN(dst.height(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	ToFloatImageKernel<<<blocks, threads>>>(
		server_, dst.server(), scale, offset);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());

	return dst;
}

template<typename T>
std::tuple<ImageCuda<typename T::VecTypef>,
    ImageCuda<typename T::VecTypef>> ImageCuda<T>::Gradient() {
	ImageCuda<typename T::VecTypef> dx;
	ImageCuda<typename T::VecTypef> dy;
	dx.Init(width_, height_);
	dy.Init(width_, height_);

	const dim3 blocks(UPPER_ALIGN(width_, THREAD_2D_UNIT),
		UPPER_ALIGN(height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	GradientImageKernel<<<blocks, threads>>>(
		server_, dx.server(), dy.server());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());

	return std::make_tuple(dx, dy);
}
}
#endif