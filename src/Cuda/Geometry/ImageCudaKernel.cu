#include "ImageCuda.cuh"

namespace three {

/**
 * Downsampling
 */
template<typename T>
__global__
void DownsampleImageKernel(ImageCudaServer<T> src, ImageCudaServer<T> dst) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	dst.get(u, v) = T(0);

	const int kernel_size = 5;
	const int kernel_size_2 = (kernel_size >> 1);

	int u_x2 = u << 1;
	int v_x2 = v << 1;

	int u_min = max(0, u_x2 - kernel_size_2);
	int v_min = max(0, v_x2 - kernel_size_2);

	int u_max = min(src.width_ - 1, u_x2 + kernel_size_2);
	int v_max = min(src.height_ - 1, v_x2 + kernel_size_2);

	auto sum_val = T::Vectorf();
	float sum_weight = 0;

	/**
	 * https://docs.opencv.org/2.4/doc/tutorials/imgproc/pyramids/pyramids.html
	 * [1 4 6 4 1] x [1 4 6 4 1]
	 */
	const float weights[3] = {0.375f, 0.25f, 0.0625f};
	for (int uu = u_min; uu <= u_max; ++uu) {
		for (int vv = v_min; vv <= v_max; ++vv) {
			auto val = src.get(uu, vv).ToVectorf();
			float weight = weights[abs(uu - u_x2)] * weights[abs(vv - v_x2)];
			sum_val += val * weight;
			sum_weight += weight;
		}
	}

	sum_val /= sum_weight;
	dst.get(u, v).FromVectorf(sum_val);
}

/**
 * Specification for depth images: DON'T mix up null values
 */
template<>
__global__
void DownsampleImageKernel<Vector1s>
(ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	dst.get(u, v) = 0;

	const int kernel_size = 5;
	const int kernel_size_2 = (kernel_size >> 1);

	int u_x2 = u << 1;
	int v_x2 = v << 1;

	int u_min = max(0, u_x2 - kernel_size_2);
	int v_min = max(0, v_x2 - kernel_size_2);

	int u_max = min(src.width_ - 1, u_x2 + kernel_size_2);
	int v_max = min(src.height_ - 1, v_x2 + kernel_size_2);

	auto sum_val = Vector1f();
	float sum_weight = 0;

	/**
	 * https://docs.opencv.org/2.4/doc/tutorials/imgproc/pyramids/pyramids.html
	 * [1 4 6 4 1]^T [1 4 6 4 1]
	 */
	const float weights[3] = {0.375f, 0.25f, 0.0625f};
	for (int uu = u_min; uu <= u_max; ++uu) {
		for (int vv = v_min; vv <= v_max; ++vv) {
			auto val = src.get(uu, vv);
			if (val[0] == 0) continue; /* NULL DEPTH */
			float weight = weights[abs(uu - u_x2)] * weights[abs(vv - v_x2)];
			sum_val += val.ToVectorf() * weight;
			sum_weight += weight;
		}
	}

	if (sum_weight > 0) {
		sum_val /= sum_weight;
		dst.get(u, v).FromVectorf(sum_val);
	}
}

template
__global__
void DownsampleImageKernel<Vector4b>(
	ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst);
template
__global__
void DownsampleImageKernel<Vector3b>(
	ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst);
template
__global__
void DownsampleImageKernel<Vector1b>(
	ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst);

template
__global__
void DownsampleImageKernel<Vector4f>(
	ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst);
template
__global__
void DownsampleImageKernel<Vector3f>(
	ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst);
template
__global__
void DownsampleImageKernel<Vector1f>(
	ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst);

/**
 * Gaussian
 */
template<typename T>
__global__
void GaussianImageKernel(
	ImageCudaServer<T> src, ImageCudaServer<T> dst, const int kernel_idx) {

	const int kernel_sizes[3] = {3, 5, 7};
	/** Some zero paddings, should be ignored during iteration **/
	const float kernel_weights[3][7] = {
		{0.25f, 0.5f, 0.25f, 0, 0, 0, 0},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0, 0},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	dst.get(u, v) = T(0);

	const int kernel_size = kernel_sizes[kernel_idx];

	const int kernel_size_2 = kernel_size >> 1;
	int u_min = max(0, u - kernel_size_2);
	int v_min = max(0, v - kernel_size_2);

	int u_max = min(src.width_ - 1,  u + kernel_size_2);
	int v_max = min(src.height_ - 1, v + kernel_size_2);

	auto sum_val = T::Vectorf();
	float sum_weight = 0;

	for (int uu = u_min; uu <= u_max; ++uu) {
		for (int vv = v_min; vv <= v_max; ++vv) {
			auto val = src.get(uu, vv).ToVectorf();
			float weight =
				kernel_weights[kernel_idx][abs(uu - u)] *
				kernel_weights[kernel_idx][abs(vv - v)];
			sum_val += val * weight;
			sum_weight += weight;
		}
	}

	sum_val /= sum_weight;
	dst.get(u, v).FromVectorf(sum_val);
}

template
__global__
void GaussianImageKernel<Vector1s>(
	ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector4b>(
	ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector3b>(
	ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector1b>(
	ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector4f>(
	ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector3f>(
	ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
	const int kernel_idx);

template
__global__
void GaussianImageKernel<Vector1f>(
	ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
	const int kernel_idx);

/**
 * Conversion
 */
template<typename T>
__global__
void ToFloatImageKernel(
	ImageCudaServer<T> src, ImageCudaServer<typename T::VecTypef> dst,
	float scale, float offset) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= dst.width_ || v >= dst.height_) return;

	dst.get(u, v) = src.get(u, v).ToVectorf() * scale + T::VecTypef(offset);
}

template
__global__
void ToFloatImageKernel<Vector1s>(
	ImageCudaServer<Vector1s>, ImageCudaServer<Vector1f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector4b>(
	ImageCudaServer<Vector4b>, ImageCudaServer<Vector4f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector3b>(
	ImageCudaServer<Vector3b>, ImageCudaServer<Vector3f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector1b>(
	ImageCudaServer<Vector1b>, ImageCudaServer<Vector1f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector4f>(
	ImageCudaServer<Vector4f>, ImageCudaServer<Vector4f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector3f>(
	ImageCudaServer<Vector3f>, ImageCudaServer<Vector3f> dst,
	float scale, float offset);

template
__global__
void ToFloatImageKernel<Vector1f>(
	ImageCudaServer<Vector1f>, ImageCudaServer<Vector1f> dst,
	float scale, float offset);

/**
 * Gradient, using Sobel operator
 * https://en.wikipedia.org/wiki/Sobel_operator
 */
template<typename T>
__global__
void GradientImageKernel(
	ImageCudaServer<T> src,
	ImageCudaServer<typename T::VecTypef> dx,
	ImageCudaServer<typename T::VecTypef> dy) {

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= src.width_ || v >= src.height_) return;

	if (u == 0 || u == src.width_ - 1 || v == 0 || v == src.height_ - 1) {
		dx.get(u, v) = T::VecTypef(0);
		dy.get(u, v) = T::VecTypef(0);
		return;
	}

	dx.get(u, v) = T::VecTypef();

	auto Iumvm = src.get(u - 1, v - 1).ToVectorf();
	auto Iumv0 = src.get(u - 1, v).ToVectorf();
	auto Iumvp = src.get(u - 1, v + 1).ToVectorf();
	auto Iu0vm = src.get(u, v - 1).ToVectorf();
	auto Iu0vp = src.get(u, v + 1).ToVectorf();
	auto Iupvm = src.get(u + 1, v - 1).ToVectorf();
	auto Iupv0 = src.get(u + 1, v).ToVectorf();
	auto Iupvp = src.get(u + 1, v + 1).ToVectorf();

	dx.get(u, v) = (Iupvm - Iumvm) + (Iupv0 - Iumv0) * 2 + (Iupvp - Iumvp);
	dy.get(u, v) = (Iumvp - Iumvm) + (Iu0vp - Iu0vm) * 2 + (Iupvp - Iupvm);
}

template
__global__
void GradientImageKernel<Vector1s>(
	ImageCudaServer<Vector1s> src,
	ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy);

template
__global__
void GradientImageKernel<Vector4b>(
	ImageCudaServer<Vector4b> src,
	ImageCudaServer<Vector4f> dx, ImageCudaServer<Vector4f> dy);

template
__global__
void GradientImageKernel<Vector3b>(
	ImageCudaServer<Vector3b> src,
	ImageCudaServer<Vector3f> dx, ImageCudaServer<Vector3f> dy);

template
__global__
void GradientImageKernel<Vector1b>(
	ImageCudaServer<Vector1b> src,
	ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy);

template
__global__
void GradientImageKernel<Vector4f>(
	ImageCudaServer<Vector4f> src,
	ImageCudaServer<Vector4f> dx, ImageCudaServer<Vector4f> dy);

template
__global__
void GradientImageKernel<Vector3f>(
	ImageCudaServer<Vector3f> src,
	ImageCudaServer<Vector3f> dx, ImageCudaServer<Vector3f> dy);

template
__global__
void GradientImageKernel<Vector1f>(
	ImageCudaServer<Vector1f> src,
	ImageCudaServer<Vector1f> dx, ImageCudaServer<Vector1f> dy);
}