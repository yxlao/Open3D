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
template<typename VecType>
__device__
VecType &ImageCudaServer<VecType>::get(int x, int y) {
	assert(x >= 0 && x < width_);
	assert(y >= 0 && y < height_);
	VecType *value = (VecType *) ((char *) data_ + y * pitch_) + x;
	return (*value);
}

template<typename VecType>
__device__
VecType &ImageCudaServer<VecType>::operator()(int x, int y) {
	return get(x, y);
}

/**
 * Naive interpolation without considering holes in depth images.
 */
template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::get_interp(float x, float y) {
	assert(x >= 0 && x <= width_ - 2);
	assert(y >= 0 && y <= height_ - 2);

	int x0 = (int) floor(x), y0 = (int) floor(y);
	float a = x - x0, b = y - y0;
	return (1 - a) * (1 - b) * get(x0, y0)
		+ (1 - a) * b * get(x0, y0 + 1)
		+ a * b * get(x0 + 1, y0 + 1)
		+ a * (1 - b) * get(x0 + 1, y0);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BoxFilter2x2(int x, int y) {
	int xp1 = min(width_ - 1, x + 1);
	int yp1 = min(height_ - 1, y + 1);

	auto sum_val = VecType::Vectorf();
	sum_val += get(x, y).ToVectorf();
	sum_val += get(x, yp1).ToVectorf();
	sum_val += get(xp1, y).ToVectorf();
	sum_val += get(xp1, yp1).ToVectorf();

	sum_val *= 0.25f;
	return VecType(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BoxFilter2x2WithHoles(int x, int y) {
	int xp1 = min(width_ - 1, x + 1);
	int yp1 = min(height_ - 1, y + 1);

	auto sum_val = VecType::Vectorf();
	float cnt = 0.0;
	VecType val;
	VecType zero = VecType(0);

	val = get(x, y);
	if (val != zero) { sum_val += val.ToVectorf(); cnt += 1.0f; }
	val = get(x, yp1);
	if (val != zero) { sum_val += val.ToVectorf(); cnt += 1.0f; }
	val = get(xp1, y);
	if (val != zero) { sum_val += val.ToVectorf(); cnt += 1.0f; }
	val = get(xp1, yp1);
	if (val != zero) { sum_val += val.ToVectorf(); cnt += 1.0f; }

	if (cnt == 0.0f) return zero;

	sum_val /= cnt;
	return VecType(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::GaussianFilter(int x, int y, int kernel_idx) {
	static const int kernel_sizes[3] = {3, 5, 7};
	static const float gaussian_weights[3][7] = {
		{0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	const int kernel_size = kernel_sizes[kernel_idx];
	const int kernel_size_2 = kernel_size >> 1;
	const float *kernel = gaussian_weights[kernel_idx];

	int x_min = max(0, x - kernel_size_2);
	int y_min = max(0, y - kernel_size_2);

	int x_max = min(width_ - 1, x + kernel_size_2);
	int y_max = min(height_ - 1, y + kernel_size_2);

	auto sum_val = VecType::Vectorf();
	float sum_weight = 0;

	for (int xx = x_min; xx <= x_max; ++xx) {
		for (int yy = y_min; yy <= y_max; ++yy) {
			auto val = get(xx, yy).ToVectorf();
			float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
			sum_val += val * weight;
			sum_weight += weight;
		}
	}

	sum_val /= sum_weight;
	return VecType(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::GaussianFilterWithHoles(
	int x, int y, int kernel_idx) {
	static const int kernel_sizes[3] = {3, 5, 7};
	static const float gaussian_weights[3][7] = {
		{0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	/** If it is already a hole, leave it alone **/
	VecType zero = VecType(0);
	if (get(x, y) == zero) return zero;

	const int kernel_size = kernel_sizes[kernel_idx];
	const int kernel_size_2 = kernel_size >> 1;
	const float *kernel = gaussian_weights[kernel_idx];

	int x_min = max(0, x - kernel_size_2);
	int y_min = max(0, y - kernel_size_2);

	int x_max = min(width_ - 1, x + kernel_size_2);
	int y_max = min(height_ - 1, y + kernel_size_2);

	auto sum_val = VecType::Vectorf();
	float sum_weight = 0;

	for (int xx = x_min; xx <= x_max; ++xx) {
		for (int yy = y_min; yy <= y_max; ++yy) {
			VecType val = get(xx, yy);
			if (val == zero) continue;
			auto valf = val.ToVectorf();
			float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
			sum_val += valf * weight;
			sum_weight += weight;
		}
	}

	if (sum_weight == 0) return zero;
	sum_val /= sum_weight;
	return VecType(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BilateralFilter(
	int x, int y, int kernel_idx, float val_sigma) {
	static const int kernel_sizes[3] = {3, 5, 7};
	static const float gaussian_weights[3][7] = {
		{0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	const int kernel_size = kernel_sizes[kernel_idx];
	const int kernel_size_2 = kernel_size >> 1;
	const float *kernel = gaussian_weights[kernel_idx];

	int x_min = max(0, x - kernel_size_2);
	int y_min = max(0, y - kernel_size_2);

	int x_max = min(width_ - 1, x + kernel_size_2);
	int y_max = min(height_ - 1, y + kernel_size_2);

	auto center_val = get(x, y).ToVectorf();
	auto sum_val = VecType::Vectorf();
	float sum_weight = 0;

	for (int xx = x_min; xx <= x_max; ++xx) {
		for (int yy = y_min; yy <= y_max; ++yy) {
			auto val = get(xx, yy).ToVectorf();
			float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
			float value_diff = (val - center_val).norm() / val_sigma;
			weight *= expf(- value_diff);

			sum_val += val * weight;
			sum_weight += weight;
		}
	}

	sum_val /= sum_weight;
	return VecType(sum_val);
}

template<typename VecType>
__device__
VecType ImageCudaServer<VecType>::BilateralFilterWithHoles(
	int x, int y, int kernel_idx, float val_sigma) {
	static const int kernel_sizes[3] = {3, 5, 7};
	static const float gaussian_weights[3][7] = {
		{0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	VecType zero = VecType(0);
	if (get(x, y) == zero) return zero;

	const int kernel_size = kernel_sizes[kernel_idx];
	const int kernel_size_2 = kernel_size >> 1;
	const float *kernel = gaussian_weights[kernel_idx];

	int x_min = max(0, x - kernel_size_2);
	int y_min = max(0, y - kernel_size_2);

	int x_max = min(width_ - 1, x + kernel_size_2);
	int y_max = min(height_ - 1, y + kernel_size_2);

	auto center_valf = get(x, y).ToVectorf();
	auto sum_val = VecType::Vectorf();
	float sum_weight = 0;

	for (int xx = x_min; xx <= x_max; ++xx) {
		for (int yy = y_min; yy <= y_max; ++yy) {
			auto val = get(xx, yy);
			if (val == zero) continue;

			auto valf = val.ToVectorf();
			float weight = kernel[abs(xx - x)] * kernel[abs(yy - y)];
			float value_diff = (valf - center_valf).norm() / val_sigma;
			weight *= expf(- value_diff * value_diff);

			sum_val += valf * weight;
			sum_weight += weight;
		}
	}

	sum_val /= sum_weight;
	return VecType(sum_val);
}


template<typename VecType>
__device__
ImageCudaServer<VecType>::Grad
ImageCudaServer<VecType>::Sobel(int x, int y) {
	auto Iumvm = get(x - 1, y - 1).ToVectorf();
	auto Iumv0 = get(x - 1, y).ToVectorf();
	auto Iumvp = get(x - 1, y + 1).ToVectorf();
	auto Iu0vm = get(x, y - 1).ToVectorf();
	auto Iu0vp = get(x, y + 1).ToVectorf();
	auto Iupvm = get(x + 1, y - 1).ToVectorf();
	auto Iupv0 = get(x + 1, y).ToVectorf();
	auto Iupvp = get(x + 1, y + 1).ToVectorf();

	return {
		(Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp),
		(Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm)
	};
}

/**
 * Client end
 */
template<typename VecType>
ImageCuda<VecType>::ImageCuda() : width_(-1), height_(-1) {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("Default ImageCuda constructor.\n");
#endif
}

template<typename VecType>
ImageCuda<VecType>::ImageCuda(const ImageCuda<VecType> &other) {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("ImageCuda copy constructor.\n");
#endif
	server_ = other.server();
	width_ = other.width();
	height_ = other.height();
	pitch_ = other.pitch();
	if (server_.ref_count_ != nullptr) {
		(*server_.ref_count_)++;
#ifdef __TRACE_LIFE_CYCLE__
		PrintInfo("ref count after copy construction: %d\n",
				*(server_.ref_count_));
#endif
	}
}

template<typename VecType>
ImageCuda<VecType> &ImageCuda<VecType>::operator=(const ImageCuda<VecType> &other) {
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
			(*server_.ref_count_)++;
#ifdef __TRACE_LIFE_CYCLE__
			PrintInfo("ref count after assignment: %d\n", *server_.ref_count_);
#endif
		}
	}

	return *this;
}

template<typename VecType>
ImageCuda<VecType>::~ImageCuda() {
#ifdef __TRACE_LIFE_CYCLE__
	PrintInfo("Destructor.\n");
#endif
	Release();
}

template<typename VecType>
int ImageCuda<VecType>::Resize(int width, int height) {
	assert(width > 0 && height > 0);
	if (width == width_ && height == height_) {
		PrintWarning("No need to resize!\n");
		return 1;
	}

	Release();
	return Create(width, height);
}

template<typename VecType>
int ImageCuda<VecType>::Create(int width, int height) {
	assert(width > 0 && height > 0);

	if (server_.ref_count_ != nullptr) {
		PrintWarning("Already created, re-creating!\n");
		return -1;
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
							  sizeof(VecType) * width_, height_));
	pitch_ = (int) pitch_size_t;
	server_.width_ = width_;
	server_.height_ = height_;
	server_.pitch_ = pitch_;

	return 0;
}

template<typename VecType>
void ImageCuda<VecType>::Release() {
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
	pitch_ = -1;
}

template<typename VecType>
void ImageCuda<VecType>::CopyTo(ImageCuda<VecType> &other) const {
	if (this == &other) return;

	if (other.server().ref_count_ == nullptr) {
		other.Create(width_, height_);
	}

	if (other.width() != width_ || other.height() != height_) {
		PrintError("Incompatible image size!\n");
		return;
	}

	CheckCuda(cudaMemcpy2D(other.server().data(), other.pitch(),
						   server_.data_, pitch_,
						   sizeof(VecType) * width_, height_,
						   cudaMemcpyDeviceToDevice));
}

template<typename VecType>
int ImageCuda<VecType>::Upload(cv::Mat &m) {
	assert(m.rows > 0 && m.cols > 0);

	/* Type checking */
	if (typeid(VecType) == typeid(Vector1s)) {
		assert(m.type() == CV_16UC1);
	} else if (typeid(VecType) == typeid(Vector4b)) {
		assert(m.type() == CV_8UC4);
	} else if (typeid(VecType) == typeid(Vector3b)) {
		assert(m.type() == CV_8UC3);
	} else if (typeid(VecType) == typeid(Vector1b)) {
		assert(m.type() == CV_8UC1);
	} else if (typeid(VecType) == typeid(Vector4f)) {
		assert(m.type() == CV_32FC4);
	} else if (typeid(VecType) == typeid(Vector3f)) {
		assert(m.type() == CV_32FC3);
	} else if (typeid(VecType) == typeid(Vector1f)) {
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
						   sizeof(VecType) * m.cols, m.rows,
						   cudaMemcpyHostToDevice));
	return 0;
}

template<typename VecType>
cv::Mat ImageCuda<VecType>::Download() {
	cv::Mat m;
	if (typeid(VecType) == typeid(Vector1s)) {
		m = cv::Mat(height_, width_, CV_16UC1);
	} else if (typeid(VecType) == typeid(Vector4b)) {
		m = cv::Mat(height_, width_, CV_8UC4);
	} else if (typeid(VecType) == typeid(Vector3b)) {
		m = cv::Mat(height_, width_, CV_8UC3);
	} else if (typeid(VecType) == typeid(Vector1b)) {
		m = cv::Mat(height_, width_, CV_8UC1);
	} else if (typeid(VecType) == typeid(Vector4f)) {
		m = cv::Mat(height_, width_, CV_32FC4);
	} else if (typeid(VecType) == typeid(Vector3f)) {
		m = cv::Mat(height_, width_, CV_32FC3);
	} else if (typeid(VecType) == typeid(Vector1f)) {
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
						   sizeof(VecType) * width_, height_,
						   cudaMemcpyDeviceToHost));
	return m;
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Downsample(DownsampleMethod method) {
	ImageCuda<VecType> dst;
	dst.Create(width_ / 2, height_ / 2);
	Downsample(dst, method);
	return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Downsample(ImageCuda<VecType> &image,
									DownsampleMethod method) {
	if (image.server().ref_count_ == nullptr
		|| image.width() != width_ / 2 || image.height() != height_ / 2) {
		image.Resize(width_ / 2, height_ / 2);
	}

	const dim3 blocks(UPPER_ALIGN(image.width(), THREAD_2D_UNIT),
					  UPPER_ALIGN(image.height(), THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	DownsampleImageKernel << < blocks, threads >> > (server_, image.server(),
		method);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

template<typename VecType>
std::tuple<ImageCuda<typename VecType::VecTypef>,
		   ImageCuda<typename VecType::VecTypef>> ImageCuda<VecType>::Sobel() {
	ImageCuda<typename VecType::VecTypef> dx;
	ImageCuda<typename VecType::VecTypef> dy;
	dx.Create(width_, height_);
	dy.Create(width_, height_);
	Sobel(dx, dy);
	return std::make_tuple(dx, dy);
}

template<typename VecType>
void ImageCuda<VecType>::Sobel(ImageCuda<typename VecType::VecTypef> &dx,
							   ImageCuda<typename VecType::VecTypef> &dy) {
	if (dx.server().ref_count_ == nullptr
		|| dx.width() != width_ || dx.height() != height_
		|| dy.server().ref_count_ == nullptr
		|| dy.width() != width_ || dy.height() != height_) {
		dx.Resize(width_, height_);
		dy.Resize(width_, height_);
	}

	const dim3 blocks(UPPER_ALIGN(width_, THREAD_2D_UNIT),
					  UPPER_ALIGN(height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	SobelImageKernel << < blocks, threads >> > (
		server_, dx.server(), dy.server());
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());

	return;
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Gaussian(GaussianKernelSize kernel,
												bool with_holes) {
	ImageCuda<VecType> dst;
	dst.Create(width_, height_);
	Gaussian(dst, kernel, with_holes);
	return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Gaussian(ImageCuda<VecType> &image,
								  GaussianKernelSize kernel,
								  bool with_holes) {
	if (image.server().ref_count_ == nullptr
		|| image.width() != width_ || image.height() != height_) {
		image.Resize(width_, height_);
	}

	const dim3 blocks(UPPER_ALIGN(width_, THREAD_2D_UNIT),
					  UPPER_ALIGN(height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	GaussianImageKernel << < blocks, threads >> > (
		server_, image.server(), (int) kernel, with_holes);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

template<typename VecType>
ImageCuda<VecType> ImageCuda<VecType>::Bilateral(GaussianKernelSize kernel,
												 float val_sigma,
												 bool with_holes) {
	ImageCuda<VecType> dst;
	dst.Create(width_, height_);
	Bilateral(dst, kernel, val_sigma, with_holes);
	return dst;
}

template<typename VecType>
void ImageCuda<VecType>::Bilateral(ImageCuda<VecType> &image,
								   GaussianKernelSize kernel,
								   float val_sigma,
								   bool with_holes) {
	if (image.server().ref_count_ == nullptr
		|| image.width() != width_ || image.height() != height_) {
		image.Resize(width_, height_);
	}

	const dim3 blocks(UPPER_ALIGN(width_, THREAD_2D_UNIT),
					  UPPER_ALIGN(height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	BilateralImageKernel<< < blocks, threads >> > (
		server_, image.server(), (int) kernel, val_sigma, with_holes);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}

template<typename VecType>
ImageCuda<typename VecType::VecTypef> ImageCuda<VecType>::ToFloat(
	float scale, float offset) {
	ImageCuda<typename VecType::VecTypef> dst;
	dst.Create(width_, height_);
	ToFloat(dst, scale, offset);
	return dst;
}

template<typename VecType>
void ImageCuda<VecType>::ToFloat(ImageCuda<typename VecType::VecTypef> &image,
								 float scale, float offset) {
	if (image.server().ref_count_ == nullptr
		|| image.width() != width_ || image.height() != height_) {
		image = ToFloat(scale, offset);
	}

	const dim3 blocks(UPPER_ALIGN(width_, THREAD_2D_UNIT),
					  UPPER_ALIGN(height_, THREAD_2D_UNIT));
	const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
	ToFloatImageKernel << < blocks, threads >> > (
		server_, image.server(), scale, offset);
	CheckCuda(cudaDeviceSynchronize());
	CheckCuda(cudaGetLastError());
}
}
#endif