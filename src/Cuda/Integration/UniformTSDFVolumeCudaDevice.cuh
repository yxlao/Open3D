//
// Created by wei on 10/9/18.
//

#pragma once

#include "UniformTSDFVolumeCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Geometry/ImageCudaDevice.cuh>

namespace open3d {
namespace cuda {
/**
 * Server end
 */
/** Coordinate conversions **/
template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaDevice<N>::InVolume(const Vector3i &X) {
    return 0 <= X(0) && X(0) < (N - 1)
        && 0 <= X(1) && X(1) < (N - 1)
        && 0 <= X(2) && X(2) < (N - 1);
}

template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaDevice<N>::InVolumef(const Vector3f &X) {
    return 0 <= X(0) && X(0) < (N - 1)
        && 0 <= X(1) && X(1) < (N - 1)
        && 0 <= X(2) && X(2) < (N - 1);
}

template<size_t N>
__device__
inline Vector3f
UniformTSDFVolumeCudaDevice<N>::world_to_voxelf(
    const Vector3f &Xw) {
    return volume_to_voxelf(transform_world_to_volume_ * Xw);
}
template<size_t N>
__device__
inline Vector3f
UniformTSDFVolumeCudaDevice<N>::voxelf_to_world(
    const Vector3f &X) {
    return transform_volume_to_world_ * voxelf_to_volume(X);
}

template<size_t N>
__device__
inline Vector3f
UniformTSDFVolumeCudaDevice<N>::voxelf_to_volume(
    const Vector3f &X) {
    return Vector3f((X(0) + 0.5f) * voxel_length_,
                    (X(1) + 0.5f) * voxel_length_,
                    (X(2) + 0.5f) * voxel_length_);
}

template<size_t N>
__device__
inline Vector3f
UniformTSDFVolumeCudaDevice<N>::volume_to_voxelf(
    const Vector3f &Xv) {
    return Vector3f(Xv(0) * inv_voxel_length_ - 0.5f,
                    Xv(1) * inv_voxel_length_ - 0.5f,
                    Xv(2) * inv_voxel_length_ - 0.5f);
}

template<size_t N>
__device__
    Vector3f
UniformTSDFVolumeCudaDevice<N>::gradient(const Vector3i &X) {
    Vector3f
    n = Vector3f::Zeros();
    Vector3i
    X1 = X, X0 = X;

#pragma unroll 1
    for (size_t k = 0; k < 3; ++k) {
        X1(k) = O3D_MIN(X(k) + 1, int(N) - 1);
        X0(k) = O3D_MAX(X(k) - 1, 0);
        n(k) = tsdf_[IndexOf(X1)] - tsdf_[IndexOf(X0)];
        X1(k) = X0(k) = X(k);
    }
    return n;
}

/** Interpolations. **/
/** Ensure it is called within [0, N - 1)^3 **/
template<size_t N>
__device__
float UniformTSDFVolumeCudaDevice<N>::TSDFAt(const Vector3f &X) {
    Vector3i Xi = X.template cast<int>();
    Vector3f r = X - Xi.template cast<float>();

    return (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf_[IndexOf(Xi + Vector3i(0, 0, 0))] +
                r(2) * tsdf_[IndexOf(Xi + Vector3i(0, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * tsdf_[IndexOf(Xi + Vector3i(0, 1, 0))] +
                r(2) * tsdf_[IndexOf(Xi + Vector3i(0, 1, 1))]
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf_[IndexOf(Xi + Vector3i(1, 0, 0))] +
                r(2) * tsdf_[IndexOf(Xi + Vector3i(1, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * tsdf_[IndexOf(Xi + Vector3i(1, 1, 0))] +
                r(2) * tsdf_[IndexOf(Xi + Vector3i(1, 1, 1))]
        ));
}

template<size_t N>
__device__
    uchar
UniformTSDFVolumeCudaDevice<N>::WeightAt(const Vector3f &X) {
    Vector3i
    Xi = X.template cast<int>();
    Vector3f
    r = X - Xi.template cast<float>();

    return uchar((1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight_[IndexOf(Xi + Vector3i(0, 0, 0))] +
                r(2) * weight_[IndexOf(Xi + Vector3i(0, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * weight_[IndexOf(Xi + Vector3i(0, 1, 0))] +
                r(2) * weight_[IndexOf(Xi + Vector3i(0, 1, 1))]
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight_[IndexOf(Xi + Vector3i(1, 0, 0))] +
                r(2) * weight_[IndexOf(Xi + Vector3i(1, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * weight_[IndexOf(Xi + Vector3i(1, 1, 0))] +
                r(2) * weight_[IndexOf(Xi + Vector3i(1, 1, 1))]
        )));
}

template<size_t N>
__device__
    Vector3b
UniformTSDFVolumeCudaDevice<N>::ColorAt(const Vector3f &X) {
    Vector3i
    Xi = X.template cast<int>();
    Vector3f
    r = X - Xi.template cast<float>();

    Vector3f
    colorf = (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(0, 0, 0))].template cast<float>() +
                r(2) * color_[IndexOf(Xi + Vector3i(0, 0, 1))].template cast<float>()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(0, 1, 0))].template cast<float>() +
                r(2) * color_[IndexOf(Xi + Vector3i(0, 1, 1))].template cast<float>()
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(1, 0, 0))].template cast<float>() +
                r(2) * color_[IndexOf(Xi + Vector3i(1, 0, 1))].template cast<float>()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(1, 1, 0))].template cast<float>() +
                r(2) * color_[IndexOf(Xi + Vector3i(1, 1, 1))].template cast<float>()
        ));

    return colorf.template saturate_cast<uchar>();
}

template<size_t N>
__device__
    Vector3f
UniformTSDFVolumeCudaDevice<N>::GradientAt(const Vector3f &X) {
    Vector3f
    n = Vector3f::Zeros();

    const float half_gap = voxel_length_;
    const float epsilon = 0.1f * voxel_length_;
    Vector3f
    X0 = X, X1 = X;

#pragma unroll 1
    for (size_t k = 0; k < 3; k++) {
        X0(k) = fmaxf(X0(k) - half_gap, epsilon);
        X1(k) = fminf(X1(k) + half_gap, N - 1 - epsilon);
        n(k) = (TSDFAt(X1) - TSDFAt(X0));

        X0(k) = X1(k) = X(k);
    }
    return n;
}

/** High level methods **/
template<size_t N>
__device__
void UniformTSDFVolumeCudaDevice<N>::Integrate(
    const Vector3i &X,
    RGBDImageCudaDevice &rgbd,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    /** Projective data association **/
    Vector3f
    Xw = voxelf_to_world(X.template cast<float>());
    Vector3f
    Xc = transform_camera_to_world.Inverse() * Xw;
    Vector2f p = camera.ProjectPoint(Xc);

    /** TSDF **/
    if (!camera.IsPixelValid(p)) return;
    float d = rgbd.depth_.interp_at(p(0), p(1))(0);

    float tsdf = d - Xc(2);
    if (tsdf <= -sdf_trunc_) return;
    tsdf = fminf(tsdf, sdf_trunc_);

    Vector3b color = rgbd.color_raw_.at(int(p(0)), int(p(1)));

    float &tsdf_sum = this->tsdf(X);
    uchar & weight_sum = this->weight(X);
    Vector3b & color_sum = this->color(X);

    float w0 = 1 / (weight_sum + 1.0f);
    float w1 = 1 - w0;

    tsdf_sum = tsdf * w0 + tsdf_sum * w1;
    color_sum = Vector3b(color(0) * w0 + color_sum(0) * w1,
                         color(1) * w0 + color_sum(1) * w1,
                         color(2) * w0 + color_sum(2) * w1);
    weight_sum = uchar(fminf(weight_sum + 1.0f, 255));
}

template<size_t N>
__device__
    Vector3f
UniformTSDFVolumeCudaDevice<N>::RayCasting(
    const Vector2i &p,
    PinholeCameraIntrinsicCuda &camera,
    TransformCuda &transform_camera_to_world) {

    Vector3f
    ret = Vector3f(0);

    Vector3f
    ray_c = camera.InverseProjectPixel(p, 1.0f).normalized();

    /** TODO: throw it into parameters **/
    const float t_min = 0.2f / ray_c(2);
    const float t_max = 3.5f / ray_c(2);

    const Vector3f camera_origin_v = transform_world_to_volume_ *
        (transform_camera_to_world * Vector3f::Zeros());
    const Vector3f ray_v = transform_world_to_volume_.Rotate(
        transform_camera_to_world.Rotate(ray_c));

    float t_prev = 0, tsdf_prev = 0;

    /** Do NOT use #pragma unroll: it will make it slow **/
    float t_curr = t_min;
    while (t_curr < t_max) {
        Vector3f
        Xv_t = camera_origin_v + t_curr * ray_v;
        Vector3f
        X_t = volume_to_voxelf(Xv_t);

        if (!InVolumef(X_t)) return ret;

        float tsdf_curr = this->tsdf(X_t.template cast<int>());

        float step_size = tsdf_curr == 0 ?
                          sdf_trunc_ : fmaxf(tsdf_curr, voxel_length_);

        if (tsdf_prev > 0 && tsdf_curr < 0) { /** Zero crossing **/
            float t_intersect = (t_curr * tsdf_prev - t_prev * tsdf_curr)
                / (tsdf_prev - tsdf_curr);

            Vector3f
            Xv_surface_t = camera_origin_v + t_intersect * ray_v;
            Vector3f
            X_surface_t = volume_to_voxelf(Xv_surface_t);
            Vector3f
            normal_v_t = GradientAt(X_surface_t).normalized();

            return transform_camera_to_world.Inverse().Rotate(
                transform_volume_to_world_.Rotate(normal_v_t)).normalized();
        }

        tsdf_prev = tsdf_curr;
        t_prev = t_curr;
        t_curr += step_size;
    }

    return ret;
}
} // cuda
} // open3d