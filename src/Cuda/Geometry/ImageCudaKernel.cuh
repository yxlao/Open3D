#include "ImageCuda.cuh"

namespace three {

/**
 * Downsampling
 */
template<typename VecType>
__global__
void DownsampleImageKernel(ImageCudaServer<VecType> src,
						   ImageCudaServer<VecType> dst,
						   DownsampleMethod method) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	/** Re-write the function with specific options if we want to accelerate **/
	switch (method) {
		case BoxFilter:
			dst.get(u, v) = src.BoxFilter2x2(u << 1, v << 1);
			return;
		case BoxFilterWithHoles:
			dst.get(u, v) = src.BoxFilter2x2WithHoles(u << 1, v << 1);
			return;
		case GaussianFilter:
			dst.get(u, v) = src.GaussianFilter(u << 1, v << 1, Gaussian5x5);
			return;
		case GaussianFilterWithHoles:
			dst.get(u, v) = src.GaussianFilterWithHoles(u << 1, v << 1,
														Gaussian5x5);
			return;
		default: printf("Unsupported method.\n");
	}
}

template
__global__
void DownsampleImageKernel<Vector1s>(
	ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
	DownsampleMethod method);

template
__global__
void DownsampleImageKernel<Vector4b>(
	ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
	DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector3b>(
	ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
	DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector1b>(
	ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
	DownsampleMethod method);

template
__global__
void DownsampleImageKernel<Vector4f>(
	ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
	DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector3f>(
	ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
	DownsampleMethod method);
template
__global__
void DownsampleImageKernel<Vector1f>(
	ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
	DownsampleMethod method);

/**
 * Gaussian
 */
template<typename VecType>
__global__
void GaussianImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<VecType> dst,
	const int kernel_idx) {

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	dst.get(u, v) = src.GaussianFilterWithHoles(u, v, kernel_idx);
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

template<typename VecType>
__global__
void BilateralImageKernel(ImageCudaServer<VecType> src,
						  ImageCudaServer<VecType> dst,
						  const int kernel_idx,
						  float val_sigma) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	dst.get(u, v) = src.BilateralFilter(u, v, kernel_idx, val_sigma);
}

template
__global__
void BilateralImageKernel<Vector1s>(
	ImageCudaServer<Vector1s> src, ImageCudaServer<Vector1s> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector4b>(
	ImageCudaServer<Vector4b> src, ImageCudaServer<Vector4b> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector3b>(
	ImageCudaServer<Vector3b> src, ImageCudaServer<Vector3b> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector1b>(
	ImageCudaServer<Vector1b> src, ImageCudaServer<Vector1b> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector4f>(
	ImageCudaServer<Vector4f> src, ImageCudaServer<Vector4f> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector3f>(
	ImageCudaServer<Vector3f> src, ImageCudaServer<Vector3f> dst,
	const int kernel_idx, float val_sigma);

template
__global__
void BilateralImageKernel<Vector1f>(
	ImageCudaServer<Vector1f> src, ImageCudaServer<Vector1f> dst,
	const int kernel_idx, float val_sigma);

/**
 * Conversion
 */
template<typename VecType>
__global__
void ToFloatImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<typename VecType::VecTypef> dst,
	float scale,
	float offset) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= dst.width_ || v >= dst.height_) return;

	dst.get(u, v) =
		src.get(u, v).ToVectorf() * scale + VecType::VecTypef(offset);
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
template<typename VecType>
__global__
void GradientImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<typename VecType::VecTypef> dx,
	ImageCudaServer<typename VecType::VecTypef> dy) {

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= src.width_ || v >= src.height_) return;

	if (u == 0 || u == src.width_ - 1 || v == 0 || v == src.height_ - 1) {
		dx.get(u, v) = VecType::VecTypef(0);
		dy.get(u, v) = VecType::VecTypef(0);
		return;
	}

	typename ImageCudaServer<VecType>::Grad grad = src.Gradient(u, v);
	dx(u, v) = grad.dx;
	dy(u, v) = grad.dy;
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