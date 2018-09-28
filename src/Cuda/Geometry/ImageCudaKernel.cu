#include "ImageCuda.cuh"

namespace three {

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
}