//
// Created by wei on 10/1/18.
//

#ifndef OPEN3D_RGBDODOMETRYCUDA_C_H
#define OPEN3D_RGBDODOMETRYCUDA_C_H

#include "RGBDOdometryCuda.h"
#include <Cuda/Geometry/ImageCuda.cuh>
#include <Cuda/Container/ArrayCuda.cuh>
#include <sophus/se3.hpp>

namespace three {

/**
 * Server end
 */
template<size_t N>
__device__
bool RGBDOdometryCudaServer<N>::ComputePixelwiseJacobiansAndResiduals(
	int x, int y, size_t level,
	three::JacobianCuda<6> &jacobian_I,
	three::JacobianCuda<6> &jacobian_D,
	float &residual_I,
	float &residual_D) {

	/** Check 1: depth valid in source? **/
	float d_source = source_depth(level).get(x, y)(0);
	bool mask = IsValidDepth(d_source);
	if (!mask) return false;

	/** Check 2: reprojected point in image? **/
	Vector3f
	X = transform_source_to_target_
		* pinhole_camera_intrinsics_.InverseProjection(x, y, d_source, level);

	Vector2f p_warped = pinhole_camera_intrinsics_.Projection(X, level);
	mask = pinhole_camera_intrinsics_.IsValid(p_warped, level);
	if (!mask) return false;

	/** Check 3: depth valid in target? Occlusion? **/
	float d_target = target_depth(level).get_interp_with_holes(
		p_warped(0), p_warped(1))(0);
	mask = IsValidDepth(d_target) && IsValidDepthDiff(d_target - X(2));
	if (!mask) return false;

	/** Checks passed, let's rock!
	 *  \partial D(p_warped) \partial p_warped: [dx_D, dy_D] at p_warped, 1x2
	 *  \partial I(p_warped) \partial p_warped: [dx_I, dy_I] at p_warped, 1x2
	 *  \partial X.z \partial X: [0, 0, 1], 1x3
	 *  \partial p_warped \partial X: [fx/Z, 0, -fx X/Z^2;
	 *                                 0, fy/Z, -fy Y/Z^2]            2x3
	 *  \partial X \partial \xi: [I | -[X]^] = [1 0 0 0  Z -Y;
	 *                                          0 1 0 -Z 0 X;
	 *                                          0 0 1 Y -X 0]         3x6
	 * J_I = (d I(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
	 * J_D = (d D(p_warped) / d p_warped) (d p_warped / d X) (d X / d \xi)
	 *     - (d X.z / d X) (d X / d \xi)
	 */
	float dx_I = target_intensity_dx(level).get_interp(
		p_warped(0), p_warped(1))(0);
	float dy_I = target_intensity_dy(level).get_interp(
		p_warped(0), p_warped(1))(0);
	float dx_D = target_depth_dx(level).get_interp(
		p_warped(0), p_warped(1))(0);
	float dy_D = target_depth_dy(level).get_interp(
		p_warped(0), p_warped(1))(0);
	float fx = pinhole_camera_intrinsics_.fx(level);
	float fy = pinhole_camera_intrinsics_.fy(level);
	float inv_Z = 1.0f / X(2);
	float fx_on_Z = fx * inv_Z;
	float fy_on_Z = fy * inv_Z;

	float c0 = dx_I * fx_on_Z;
	float c1 = dy_I * fy_on_Z;
	float c2 = -(c0 * X(0) + c1 * X(1)) * inv_Z;
	jacobian_I(0) = sqrt_coeff_I_ * c0;
	jacobian_I(1) = sqrt_coeff_I_ * c1;
	jacobian_I(2) = sqrt_coeff_I_ * c2;
	jacobian_I(3) = sqrt_coeff_I_ * (-X(2) * c1 + X(1) * c2);
	jacobian_I(4) = sqrt_coeff_I_ * (X(2) * c0 - X(0) * c2);
	jacobian_I(5) = sqrt_coeff_I_ * (-X(1) * c0 + X(0) * c1);
	residual_I = sqrt_coeff_I_ *
		source_intensity(level).get_interp(p_warped(0), p_warped(1))(0)
		- target_intensity(level).get(x, y)(0);

	float d0 = dx_D * fx_on_Z;
	float d1 = dy_D * fy_on_Z;
	float d2 = -(d0 * X(0) + d1 * X(1)) * inv_Z;
	jacobian_D(0) = sqrt_coeff_D_ * d0;
	jacobian_D(1) = sqrt_coeff_D_ * d1;
	jacobian_D(2) = sqrt_coeff_D_ * (d2 - 1.0f);
	jacobian_D(3) = sqrt_coeff_D_ * ((-X(2) * d1 + X(1) * d2) - X(1));
	jacobian_D(4) = sqrt_coeff_D_ * ((X(2) * d0 - X(0) * d2) + X(0));
	jacobian_D(5) = sqrt_coeff_D_ * (-X(1) * d0 + X(0) * d1);
	residual_D = sqrt_coeff_D_ * (d_source - X(2));

	return true;
}

template<size_t N>
__device__
bool RGBDOdometryCudaServer<N>::ComputePixelwiseJtJAndJtr(
	JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
	float &residual_I, float &residual_D,
	HessianCuda<6> &JtJ, Vector6f &Jtr) {
	JtJ = jacobian_I.ComputeJtJ() + jacobian_D.ComputeJtJ();
	Jtr = jacobian_I.ComputeJtr(residual_I) + jacobian_D.ComputeJtr(residual_D);

	return true;
}

/**
 * Client end
 */
template<size_t N>
RGBDOdometryCuda<N>::RGBDOdometryCuda() {}

template<size_t N>
RGBDOdometryCuda<N>::~RGBDOdometryCuda() {
	Release();
}

template<size_t N>
void RGBDOdometryCuda<N>::Create(int width, int height) {
	target_depth_.Create(width, height);
	target_depth_dx_.Create(width, height);
	target_depth_dy_.Create(width, height);

	target_intensity_.Create(width, height);
	target_intensity_dx_.Create(width, height);
	target_intensity_dy_.Create(width, height);

	source_depth_.Create(width, height);
	source_intensity_.Create(width, height);

	results_.Create(28);
}

template<size_t N>
void RGBDOdometryCuda<N>::Release() {
	target_depth_.Release();
	target_depth_dx_.Release();
	target_depth_dy_.Release();

	target_intensity_.Release();
	target_intensity_dx_.Release();
	target_intensity_dy_.Release();

	source_depth_.Release();
	source_intensity_.Release();
}

template<size_t N>
void RGBDOdometryCuda<N>::ExtractResults(std::vector<float> &results,
										 Eigen::Matrix<float, 6, 6> &JtJ,
										 Eigen::Matrix<float, 6, 1> &Jtr,
										 float &residual,
										 float &inliers) {
	int cnt = 0;
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			JtJ(i, j) = JtJ(j, i) = results[cnt];
			++cnt;
		}
	}
	for (int i = 0; i < 6; ++i) {
		Jtr(i) = results[cnt];
		++cnt;
	}
	residual = results[cnt];
	++cnt;
	inliers = results[cnt];
}

template<size_t N>
__host__
void RGBDOdometryCuda<N>::Apply(ImageCuda<Vector1f> &source_depth,
								ImageCuda<Vector1f> &source_intensity,
								ImageCuda<Vector1f> &target_depth,
								ImageCuda<Vector1f> &target_intensity) {
	source_depth_.Build(source_depth);
	server_.source_depth_ = source_depth_.server();

	source_intensity_.Build(source_depth);
	server_.source_intensity_ = source_intensity_.server();

	target_depth_.Build(target_depth);
	server_.target_depth_ = target_depth_.server();

	target_intensity_.Build(target_depth);
	server_.target_intensity_ = target_intensity_.server();

	for (size_t level = 0; level < N; ++level) {
		target_depth_.get(level).Sobel(
			target_depth_dx_.get(level),
			target_depth_dy_.get(level));
		target_intensity_.get(level).Sobel(
			target_intensity_dx_.get(level),
			target_intensity_dy_.get(level));
	}

	/** TODO: add reference count for Array **/
	results_.Create(29);

	for (int level = N - 1; level >= 0; --level) {
		server_.transform_source_to_target_.FromEigen(
			transform_source_to_target_);

		const dim3 blocks(
			UPPER_ALIGN(source_depth_.width(level), THREAD_2D_UNIT),
			UPPER_ALIGN(source_depth_.height(level), THREAD_2D_UNIT));
		const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
		ApplyRGBDOdometryKernel << <blocks, threads>>>(server_, level);
		CheckCuda(cudaDeviceSynchronize());
		CheckCuda(cudaGetLastError());

		std::vector<float> results = results_.DownloadAll();
		Matrix6f JtJ;
		Vector6f Jtr;
		float residuals;
		float inliers;
		ExtractResults(results, JtJ, Jtr, residuals, inliers);

		Vector6d dxi = JtJ.cast<double>().ldlt().solve(-Jtr.cast<double>());
		transform_source_to_target_ =
			Sophus::SE3d::exp(dxi).matrix().cast<float>() *
				transform_source_to_target_;
	}
}
}
#endif //OPEN3D_RGBDODOMETRYCUDA_C_H
