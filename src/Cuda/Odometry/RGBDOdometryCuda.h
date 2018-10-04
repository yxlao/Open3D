//
// Created by wei on 10/1/18.
//

#ifndef OPEN3D_RGBDODOMETRY_H
#define OPEN3D_RGBDODOMETRY_H

#include "OdometryClasses.h"
#include <Cuda/Container/ArrayCuda.h>
#include <Cuda/Geometry/ImagePyramidCuda.h>
#include <Cuda/Geometry/VectorCuda.h>
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
 *   (1 - sigma) ||I_{source}[g(s(h(p, D_{target}), \xi))] - I_{target}[p]||^2
 * + sigma ||D_{source}[g(s(h(p, D_{target}), \xi))] - s(h(p, D_{target})).z||^2
 *
 * Usually @source frame should be a keyframe, or N-1 frame
 *                 it should hold more precomputed information,
 *                 including gradients.
 *         @target frame should be a current frame.
 *         We warp the @target frame to the @source frame.
 */
template<size_t N>
class RGBDOdometryCudaServer {
private:
	ImagePyramidCudaServer<Vector1f, N> source_depth_;
	ImagePyramidCudaServer<Vector1f, N> source_depth_dx_;
	ImagePyramidCudaServer<Vector1f, N> source_depth_dy_;

	ImagePyramidCudaServer<Vector1f, N> source_intensity_;
	ImagePyramidCudaServer<Vector1f, N> source_intensity_dx_;
	ImagePyramidCudaServer<Vector1f, N> source_intensity_dy_;
	/* ImagePyramidCudaServer<Vector3f, N> source_normal_; */

	ImagePyramidCudaServer<Vector1f, N> target_depth_;
	ImagePyramidCudaServer<Vector1f, N> target_intensity_;
	/* ImagePyramidCudaServer<Vector3f, N> target_normal_; */

	ArrayCuda<float> results_;

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
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_depth(size_t level) {
		return source_depth_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_depth_dx(size_t level) {
		return source_depth_dx_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_depth_dy(size_t level) {
		return source_depth_dy_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_intensity(size_t level) {
		return source_intensity_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_intensity_dx(size_t level) {
		return source_intensity_dx_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_source_intensity_dy(size_t level) {
		return source_intensity_dy_.get(level);
	}

	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_target_depth(size_t level) {
		return target_depth_.get(level);
	}
	__HOSTDEVICE__ ImageCudaServer<Vector1f>& get_target_intensity(size_t level) {
		return target_intensity_.get(level);
	}

	friend class RGBDOdometryCuda<N>;
};

template<size_t N>
class RGBDOdometryCuda {
private:
	RGBDOdometryCudaServer<N> server_;

	ImagePyramidCuda<Vector1f, N> source_depth_;
	ImagePyramidCuda<Vector1f, N> source_intensity_;
	ImagePyramidCuda<Vector1f, N> target_depth_;
	ImagePyramidCuda<Vector1f, N> target_intensity_;

	ArrayCuda<float> results_;

public:
	void Create();
	void Release();

	void Apply(cv::Mat &source, cv::Mat &target);
	void Apply(ImageCuda<Vector1f>& source_depth,
			   ImageCuda<Vector1f>& source_intensity,
			   ImageCuda<Vector1f>& target_depth,
			   ImageCuda<Vector1f>& target_intensity);

	RGBDOdometryCudaServer<N>& server() {
		return server_;
	}
	const RGBDOdometryCudaServer<N>& server() const {
		return server_;
	}
};

template<size_t N>
__GLOBAL__
void RGBDOdometryKernel(RGBDOdometryCudaServer<N> odometry,
						const size_t level,
						ArrayCudaServer<float> result);

}
#endif //OPEN3D_RGBDODOMETRY_H
