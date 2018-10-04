//
// Created by wei on 10/1/18.
//

#include "RGBDOdometryCuda.cuh"

namespace three {
template<size_t N>
__global__
void RGBDOdometryKernel(RGBDOdometryCudaServer<N> odometry,
						const size_t level,
						ArrayCudaServer<float> result) {
	static __shared__ float local_sum[THREAD_2D_UNIT * THREAD_2D_UNIT];

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int tid = threadIdx.x + threadIdx.y * blockDim.x;

	bool mask = false;

	auto &target_depth = odometry.get_target_depth(level);
	auto &target_intensity = odometry.get_target_intensity(level);
	float d_target = target_depth.get(x, y)[0];
	mask = odometry.IsValidDepth(d_target);
//	if (mask) {
//		Vector2f warped_p =
//	}

	auto &source_depth = odometry.get_source_depth(level);
	auto &source_depth_dx = odometry.get_source_depth_dx(level);
	auto &source_depth_dy = odometry.get_source_depth_dy(level);
	auto &source_intensity = odometry.get_source_intensity(level);
	auto &source_intensity_dx = odometry.get_source_intensity_dx(level);
	auto &source_intensity_dy = odometry.get_source_intensity_dy(level);


	/* Reduction here */
	/*
	 * JtJ, Jtb, inlier, residual, etc
	 */
}

}