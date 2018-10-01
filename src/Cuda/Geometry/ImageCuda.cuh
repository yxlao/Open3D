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
template<typename T>
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
ImageCuda<T>::ImageCuda() : width_(-1), height_(-1) {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("Default ImageCuda constructor.\n");
#endif
}

template<typename T>
ImageCuda<T>::ImageCuda(const ImageCuda<T> &other) {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("ImageCuda copy constructor.\n");
#endif
	server_ = other.server();
	width_ = other.width();
	height_ = other.height();
	pitch_ = other.pitch();
	if (server_.ref_count_ != nullptr) {
		(*server_.ref_count_) ++;
#ifdef __TRACE_LIFE_CYCLE__
		PrintInfo("ref count after copy construction: %d\n",
				*(server_.ref_count_));
#endif
	}
}

template<typename T>
ImageCuda<T>& ImageCuda<T>::operator = (const ImageCuda<T> &other) {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("ImageCuda assignment operator.\n");
#endif
	if (this != &other) {
		Release();

		server_ = other.server();
		width_ = other.width();
		height_ = other.height();
		pitch_ = other.pitch();

		if (server_.ref_count_ != nullptr) {
			(*server_.ref_count_) ++;
#ifdef __TRACE_LIFE_CYCLE__
			PrintInfo("ref count after assignment: %d\n", *server_.ref_count_);
#endif
		}
	}

	return *this;
}

template<typename T>
ImageCuda<T>::~ImageCuda() {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("Destructor.\n");
#endif
	Release();
}

template<typename T>
int ImageCuda<T>::Create(int width, int height) {
	assert(width > 0 && height > 0);

	if (server_.ref_count_ != nullptr) {
		PrintWarning("Already created!\n");
		return 1;
	}

	width_ = width;
	height_ = height;
	pitch_ = 0;

#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("Creating.\n");
#endif

	server_.ref_count_ = new int(1);
	size_t pitch_size_t = 0;
	CheckCuda(cudaMallocPitch((void **) (&server_.data_), &pitch_size_t,
							  sizeof(T) * width_, height_));
	pitch_ = (int) pitch_size_t;
	server_.width_ = width_;
	server_.height_ = height_;
	server_.pitch_ = pitch_;

	return 0;
}

template<typename T>
void ImageCuda<T>::Release() {
#ifdef __TRACE_LIFE_CYCLE__
	if (server_.ref_count_ != nullptr) {
		PrintInfo("ref count before releasing: %d\n", *server_.ref_count_);
	}
#endif
	if (server_.ref_count_ != nullptr && --(*server_.ref_count_) == 0) {
		CheckCuda(cudaFree(server_.data_));
		delete server_.ref_count_;
		server_.ref_count_ = nullptr;
	}
	width_ = -1;
	height_ = -1;
}


template<typename T>
void ImageCuda<T>::CopyTo(ImageCuda<T> &other) const {
	if (this == &other) return;

	if (other.server().ref_count_ == nullptr) {
		other.Create(width_, height_);
	}

	if (other.width() != width_ || other.height() != height_) {
		PrintError("Incompatible image size!\n");
		return;
	}

	CheckCuda(cudaMemcpy2D(other.server().data(), other.pitch(),
		server_.data_, pitch_, sizeof(T) * width_, height_,
		cudaMemcpyDeviceToDevice));
}

template<typename T>
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
		PrintWarning("Unsupported format %d!\n");
		return -1;
	}

	if (server_.ref_count_ == nullptr) {
		Create(m.cols, m.rows);
	}
	if (width_ != m.cols || height_ != m.rows) {
		PrintWarning("Incompatible image size!\n");
		return 1;
	}

	CheckCuda(cudaMemcpy2D(server_.data_, pitch_, m.data, m.step,
		sizeof(T) * m.cols, m.rows,
		cudaMemcpyHostToDevice));
	return 0;
}

template<typename T>
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
		PrintWarning("Unsupported format %d!\n");
		return m;
	}

	if (server_.ref_count_ == nullptr) {
		PrintWarning("ImageCuda not initialized!\n");
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
	dst.Create(width_ / 2, height_ / 2);

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
void ImageCuda<T>::Downsample(ImageCuda<T> &image) {
	if (image.server().ref_count_ == nullptr
	|| image.width() != width_ / 2
	|| image.width() != height_ / 2) {
		image = Downsample();
		return;
	}

	const dim3 blocks(
		UPPER_ALIGN(image.width(), THREAD_2D_UNIT),
		UPPER_ALIGN(image.height(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	DownsampleImageKernel<<<blocks, threads>>>(server_, image.server());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

template<typename T>
ImageCuda<T> ImageCuda<T>::Gaussian(GaussianKernelOptions kernel) {
	ImageCuda<T> dst;
	dst.Create(width_, height_);

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
	dst.Create(width_, height_);

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
	dx.Create(width_, height_);
	dy.Create(width_, height_);

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