#include "ImageCuda.cuh"

namespace open3d {

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
		case BoxFilter: dst.get(u, v) = src.BoxFilter2x2(u << 1, v << 1);
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

/**
 * Shift
 */
template<typename VecType>
__global__
void ShiftImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<VecType> dst,
	float dx, float dy, bool with_holes) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	dst.get(u, v) = 0;
	if (u >= dst.width_ || v >= dst.height_)
		return;

	if (u + dx < 0 || u + dx >= src.width_ - 1
	|| v + dy < 0 || v + dy >= src.height_ - 1)
		return;

	if (with_holes) {
		dst.get(u, v) = src.get_interp_with_holes(u + dx, v + dy);
	} else {
		dst.get(u, v) = src.get_interp(u + dx, v + dy);
	}
}

/**
 * Gaussian
 */
template<typename VecType>
__global__
void GaussianImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<VecType> dst,
	const int kernel_idx,
	bool with_holes) {

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;

	if (with_holes) {
		dst.get(u, v) = src.GaussianFilterWithHoles(u, v, kernel_idx);
	} else {
		dst.get(u, v) = src.GaussianFilter(u, v, kernel_idx);
	}
}

/**
 * Bilateral
 */
template<typename VecType>
__global__
void BilateralImageKernel(ImageCudaServer<VecType> src,
						  ImageCudaServer<VecType> dst,
						  const int kernel_idx,
						  float val_sigma,
						  bool with_holes) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= dst.width_ || v >= dst.height_)
		return;
	if (with_holes) {
		dst.get(u, v) = src.BilateralFilterWithHoles(u, v, kernel_idx, val_sigma);
	} else {
		dst.get(u, v) = src.BilateralFilter(u, v, kernel_idx, val_sigma);
	}
}

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

/**
 * Gradient, using Sobel operator
 */
template<typename VecType>
__global__
void SobelImageKernel(
	ImageCudaServer<VecType> src,
	ImageCudaServer<typename VecType::VecTypef> dx,
	ImageCudaServer<typename VecType::VecTypef> dy,
	bool with_holes) {

	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= src.width_ || v >= src.height_) return;

	if (u == 0 || u == src.width_ - 1 || v == 0 || v == src.height_ - 1) {
		dx.get(u, v) = VecType::VecTypef(0);
		dy.get(u, v) = VecType::VecTypef(0);
		return;
	}

	typename ImageCudaServer<VecType>::Grad grad;
	if (with_holes) {
		grad = src.SobelWithHoles(u, v);
	} else {
		grad = src.Sobel(u, v);
	}
	dx(u, v) = grad.dx;
	dy(u, v) = grad.dy;
}
}