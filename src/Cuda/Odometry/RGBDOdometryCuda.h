//
// Created by wei on 10/1/18.
//

#ifndef OPEN3D_RGBDODOMETRY_H
#define OPEN3D_RGBDODOMETRY_H

#include "OdometryClasses.h"
#include "JacobianCuda.h"
#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
#include <Cuda/Geometry/TransformCuda.h>
#include <Cuda/Geometry/PinholeCameraCuda.h>
#include <Cuda/Common/UtilsCuda.h>
#include <Eigen/Eigen>

namespace three {
/**
 * We assume that the
 * - depths are **converted** from short
 * - intensities are **converted** from BGR / RGB ... whatever.
 *
 * Refer to this paper:
 * http://vladlen.info/papers/colored-point-cloud-registration-supplement.pdf
 *
 * We minimize
 * E(\xi) =
 * \sum_{p}
 *   (1 - sigma) ||I_{target}[g(s(h(p, D_{source}), \xi))] - I_{source}[p]||^2
 * + sigma ||D_{target}[g(s(h(p, D_{source}), \xi))] - s(h(p, D_{source})).z||^2
 *
 * Usually @target frame should be a keyframe, or N-1 frame
 *                 it should hold more precomputed information,
 *                 including gradients.
 *         @source frame should be a current frame.
 *         We warp the @source frame to the @target frame.
 */
template<size_t N>
class RGBDOdometryCudaServer {
private:
	ImagePyramidCudaServer<Vector1f, N> target_on_source_;

	ImagePyramidCudaServer<Vector1f, N> target_depth_;
	ImagePyramidCudaServer<Vector1f, N> target_depth_dx_;
	ImagePyramidCudaServer<Vector1f, N> target_depth_dy_;

	ImagePyramidCudaServer<Vector1f, N> target_intensity_;
	ImagePyramidCudaServer<Vector1f, N> target_intensity_dx_;
	ImagePyramidCudaServer<Vector1f, N> target_intensity_dy_;
	/* ImagePyramidCudaServer<Vector3f, N> source_normal_; */

	ImagePyramidCudaServer<Vector1f, N> source_depth_;
	ImagePyramidCudaServer<Vector1f, N> source_intensity_;
	/* ImagePyramidCudaServer<Vector3f, N> target_normal_; */

	ArrayCudaServer<float> results_;

public:
	PinholeCameraCuda<N> pinhole_camera_intrinsics_;
	TransformCuda transform_source_to_target_;

public:
	/** (1-sigma) * JtJ_I + sigma * JtJ_D **/
	/** To compute JtJ, we use \sqrt(1-sigma) J_I and \sqrt(sigma) J_D **/
	float sigma_;
	float sqrt_coeff_I_;
	float sqrt_coeff_D_;

public:
	float depth_near_threshold_;
	float depth_far_threshold_;
	float depth_diff_threshold_;

public:
	inline __HOSTDEVICE__ bool IsValidDepth(float depth) {
		return depth_near_threshold_ <= depth && depth <= depth_far_threshold_;
	}
	inline __HOSTDEVICE__ bool IsValidDepthDiff(float depth_diff) {
		return fabsf(depth_diff) <= depth_diff_threshold_;
	}

public:
	inline __DEVICE__ bool ComputePixelwiseJacobiansAndResiduals(
		int x, int y, size_t level,
		JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
		float &residual_I, float &residual_D);
	inline __DEVICE__ bool ComputePixelwiseJtJAndJtr(
		JacobianCuda<6> &jacobian_I, JacobianCuda<6> &jacobian_D,
		float &residual_I, float &residual_D,
		HessianCuda<6> &JtJ, Vector6f &Jtr);

public:
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_on_source() {
		return target_on_source_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_depth() {
		return target_depth_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_depth_dx() {
		return target_depth_dx_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_depth_dy() {
		return target_depth_dy_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_intensity() {
		return target_intensity_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_intensity_dx() {
		return target_intensity_dx_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	target_intensity_dy() {
		return target_intensity_dy_;
	}

	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	source_depth() {
		return source_depth_;
	}
	inline __HOSTDEVICE__ ImagePyramidCudaServer<Vector1f, N> &
	source_intensity() {
		return source_intensity_;
	}

	inline __HOSTDEVICE__ ArrayCudaServer<float> &results() {
		return results_;
	}

	friend class RGBDOdometryCuda<N>;
};

template<size_t N>
class RGBDOdometryCuda {
private:
	std::shared_ptr<RGBDOdometryCudaServer<N>> server_ = nullptr;

	ImagePyramidCuda<Vector1f, N> target_on_source_;

	ImagePyramidCuda<Vector1f, N> target_depth_;
	ImagePyramidCuda<Vector1f, N> target_depth_dx_;
	ImagePyramidCuda<Vector1f, N> target_depth_dy_;

	ImagePyramidCuda<Vector1f, N> target_intensity_;
	ImagePyramidCuda<Vector1f, N> target_intensity_dx_;
	ImagePyramidCuda<Vector1f, N> target_intensity_dy_;

	ImagePyramidCuda<Vector1f, N> source_depth_;
	ImagePyramidCuda<Vector1f, N> source_intensity_;

	ArrayCuda<float> results_;

public:
	typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f;
	typedef Eigen::Matrix<float, 6, 6, Eigen::DontAlign> Matrix6f;
	typedef Eigen::Matrix<float, 6, 1, Eigen::DontAlign> Vector6f;
	typedef Eigen::Matrix<double, 6, 1, Eigen::DontAlign> Vector6d;

public:
	Matrix4f transform_source_to_target_;

	RGBDOdometryCuda();
	~RGBDOdometryCuda();
	void SetParameters(float sigma,
		float depth_near_threshold, float depth_far_threshold,
		float depth_diff_threshold);

	void Create(int width, int height);
	void Release();
	void ConnectSubServers();

	void Build(ImageCuda<Vector1f> &source_depth,
			   ImageCuda<Vector1f> &source_intensity,
			   ImageCuda<Vector1f> &target_depth,
			   ImageCuda<Vector1f> &target_intensity);
	void Apply(ImageCuda<Vector1f> &source_depth,
			   ImageCuda<Vector1f> &source_intensity,
			   ImageCuda<Vector1f> &target_depth,
			   ImageCuda<Vector1f> &target_intensity);

	void ExtractResults(std::vector<float> &results,
						Matrix6f &JtJ, Vector6f &Jtr,
						float &error, float &inliers);

	std::shared_ptr<RGBDOdometryCudaServer<N>> &server() {
		return server_;
	}
	const std::shared_ptr<RGBDOdometryCudaServer<N>> &server() const {
		return server_;
	}
};

template<size_t N>
__GLOBAL__
void ApplyRGBDOdometryKernel(RGBDOdometryCudaServer<N> odometry,
							 size_t level);

}
#endif //OPEN3D_RGBDODOMETRY_H
