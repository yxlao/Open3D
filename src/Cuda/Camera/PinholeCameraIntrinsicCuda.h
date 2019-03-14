//
// Created by wei on 10/3/18.
//

#pragma once

#include <Open3D/Camera/PinholeCameraIntrinsic.h>
#include <Cuda/Common/LinearAlgebraCuda.h>

namespace open3d {
namespace cuda {
/**
 * This class should be passed to server, but as it is a storage class, we
 * don't name it as xxxServer
 */
class PinholeCameraIntrinsicCuda {
public:
    int width_;
    int height_;

public:
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    float inv_fx_;
    float inv_fy_;

public:
    __HOSTDEVICE__ inline int width() { return width_; }
    __HOSTDEVICE__ inline int height() { return height_; }

    __HOSTDEVICE__ PinholeCameraIntrinsicCuda() {
        width_ = -1;
        height_ = -1;
    }

    __HOSTDEVICE__ PinholeCameraIntrinsicCuda(
        int width, int height,
        float fx, float fy, float cx, float cy) {
        SetIntrinsics(width, height, fx, fy, cx, cy);
    }

    __HOST__ explicit PinholeCameraIntrinsicCuda(
        camera::PinholeCameraIntrinsic &intrinsic) {
        width_ = intrinsic.width_;
        height_ = intrinsic.height_;

        auto focal_length = intrinsic.GetFocalLength();
        fx_ = float(focal_length.first);
        inv_fx_ = 1.0f / fx_;
        fy_ = float(focal_length.second);
        inv_fy_ = 1.0f / fy_;

        auto principal_point = intrinsic.GetPrincipalPoint();
        cx_ = float(principal_point.first);
        cy_ = float(principal_point.second);
    }

    __HOSTDEVICE__ PinholeCameraIntrinsicCuda(
        camera::PinholeCameraIntrinsicParameters param) {

        if (param == camera::PinholeCameraIntrinsicParameters
        ::PrimeSenseDefault)
            SetIntrinsics(640, 480, 525.0f, 525.0f, 319.5f, 239.5f);
        else if (param == camera::PinholeCameraIntrinsicParameters::
        Kinect2DepthCameraDefault)
            SetIntrinsics(512, 424, 254.878f, 205.395f, 365.456f, 365.456f);
        else if (param == camera::PinholeCameraIntrinsicParameters::
        Kinect2ColorCameraDefault)
            SetIntrinsics(1920,
                          1080,
                          1059.9718f,
                          1059.9718f,
                          975.7193f,
                          545.9533f);
    }

    __HOSTDEVICE__ void SetIntrinsics(
        int width, int height,
        float fx, float fy, float cx, float cy) {
        width_ = width;
        height_ = height;
        fx_ = fx;
        inv_fx_ = 1.0f / fx_;
        fy_ = fy;
        inv_fy_ = 1.0f / fy_;
        cx_ = cx;
        cy_ = cy;
    }

    __HOSTDEVICE__ PinholeCameraIntrinsicCuda Downsample() {
        PinholeCameraIntrinsicCuda ret;
        ret.SetIntrinsics(width_ >> 1, height_ >> 1,
                          fx_ * 0.5f, fy_ * 0.5f, cx_ * 0.5f, cy_ * 0.5f);
        return ret;
    };

    __HOSTDEVICE__ inline bool IsPixelValid(const Vector2f &p) {
        return p(0) >= 0 && p(0) < width_ - 1
            && p(1) >= 0 && p(1) < height_ - 1;
    }

    __HOSTDEVICE__ inline bool IsPixelValid(const Vector2i &p) {
        return p(0) >= 0 && p(0) < width_ && p(1) >= 0 && p(1) < height_;
    }

    __HOSTDEVICE__ bool IsPointInFrustum(const Vector3f &X, size_t level = 0) {
        /* TODO: Derive a RGBDImage Class (using short),
         * holding depth constraints */
        if (X(2) < 0.1 || X(2) > 3) return false;
        return IsPixelValid(ProjectPoint(X));
    }

    __HOSTDEVICE__ Vector2f ProjectPoint(const Vector3f &X, size_t level = 0) {
        return Vector2f((fx_ * X(0)) / X(2) + cx_,
                        (fy_ * X(1)) / X(2) + cy_);
    }

    __HOSTDEVICE__ Vector3f InverseProjectPixel(const Vector2f &p, float d) {
        return Vector3f(d * (p(0) - cx_) * inv_fx_,
                        d * (p(1) - cy_) * inv_fy_,
                        d);
    }

    __HOSTDEVICE__ Vector3f InverseProjectPixel(const Vector2i &p, float d) {
        return Vector3f(d * (p(0) - cx_) * inv_fx_,
                        d * (p(1) - cy_) * inv_fy_,
                        d);
    }
};
} // cuda
} // open3d