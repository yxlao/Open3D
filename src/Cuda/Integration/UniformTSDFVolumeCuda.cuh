//
// Created by wei on 10/9/18.
//

#pragma once

#include "UniformTSDFVolumeCuda.h"
#include <Core/Core.h>
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Geometry/ImageCuda.cuh>

namespace open3d {

/**
 * Server end
 */
/** Coordinate conversions **/
template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaServer<N>::InVolume(const Vector3i &X) {
    return 0 <= X(0) && X(0) < (N - 1)
        && 0 <= X(1) && X(1) < (N - 1)
        && 0 <= X(2) && X(2) < (N - 1);
}

template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaServer<N>::InVolumef(const Vector3f &X) {
    return 0 <= X(0) && X(0) < (N - 1)
        && 0 <= X(1) && X(1) < (N - 1)
        && 0 <= X(2) && X(2) < (N - 1);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::world_to_voxelf(
    const Vector3f &Xw) {
    return volume_to_voxelf(transform_world_to_volume_ * Xw);
}
template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxelf_to_world(
    const Vector3f &X) {
    return transform_volume_to_world_ * voxelf_to_volume(X);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxelf_to_volume(
    const Vector3f &X) {
    return Vector3f((X(0) + 0.5f) * voxel_length_,
                    (X(1) + 0.5f) * voxel_length_,
                    (X(2) + 0.5f) * voxel_length_);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::volume_to_voxelf(
    const Vector3f &Xv) {
    return Vector3f(Xv(0) * inv_voxel_length_ - 0.5f,
                    Xv(1) * inv_voxel_length_ - 0.5f,
                    Xv(2) * inv_voxel_length_ - 0.5f);
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::gradient(const Vector3i &X) {
    Vector3f n = Vector3f::Zeros();
    Vector3i X1 = X, X0 = X;

#pragma unroll 1
    for (size_t k = 0; k < 3; ++k) {
        X1(k) = min(X(k) + 1, int(N) - 1);
        X0(k) = max(X(k) - 1, 0);
        n(k) = tsdf_[IndexOf(X1)] - tsdf_[IndexOf(X0)];
        X1(k) = X0(k) = X(k);
    }
    return n;
}

/** Interpolations. **/
/** Ensure it is called within [0, N - 1)^3 **/
template<size_t N>
__device__
float UniformTSDFVolumeCudaServer<N>::TSDFAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = X - Xi.ToVectorf();

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
uchar UniformTSDFVolumeCudaServer<N>::WeightAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = X - Xi.ToVectorf();

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
Vector3b UniformTSDFVolumeCudaServer<N>::ColorAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = X - Xi.ToVectorf();

    Vector3f colorf = (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(0, 0, 0))].ToVectorf() +
                r(2) * color_[IndexOf(Xi + Vector3i(0, 0, 1))].ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(0, 1, 0))].ToVectorf() +
                r(2) * color_[IndexOf(Xi + Vector3i(0, 1, 1))].ToVectorf()
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(1, 0, 0))].ToVectorf() +
                r(2) * color_[IndexOf(Xi + Vector3i(1, 0, 1))].ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(Xi + Vector3i(1, 1, 0))].ToVectorf() +
                r(2) * color_[IndexOf(Xi + Vector3i(1, 1, 1))].ToVectorf()
        ));

    return colorf.ToVectorb();
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::GradientAt(const Vector3f &X) {
    Vector3f n = Vector3f::Zeros();

    const float half_gap = voxel_length_;
    const float epsilon = 0.1f * voxel_length_;
    Vector3f X0 = X, X1 = X;

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
void UniformTSDFVolumeCudaServer<N>::Integrate(
    const Vector3i &X,
    ImageCudaServer<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    /** Projective data association **/
    Vector3f Xw = voxelf_to_world(X.ToVectorf());
    Vector3f Xc = transform_camera_to_world.Inverse() * Xw;
    Vector2f p = camera.Projection(Xc);

    /** TSDF **/
    if (!camera.IsValid(p)) return;
    float d = depth.get_interp(p(0), p(1))(0);

    float sdf = Xc(2) - d;
    if (sdf <= -sdf_trunc_) return;
    sdf = fminf(sdf, sdf_trunc_);

    /** Weight average **/
    /** TODO: color **/
    float &tsdf = this->tsdf(X);
    uchar &weight = this->weight(X);
    tsdf = (tsdf * weight + sdf * 1.0f) / (weight + 1.0f);
    weight = uchar(fminf(weight + 1.0f, 255));
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::RayCasting(
    const Vector2i &p,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    Vector3f ret = Vector3f(0);

    Vector3f ray_c = camera.InverseProjection(p, 1.0f).normalized();

    /** TODO: throw it into parameters **/
    const float t_min = 0.2f / ray_c(2);
    const float t_max = 3.0f / ray_c(2);

    const Vector3f camera_origin_v = transform_world_to_volume_ *
        (transform_camera_to_world * Vector3f::Zeros());
    const Vector3f ray_v = transform_world_to_volume_.Rotate(
        transform_camera_to_world.Rotate(ray_c));

    float t_prev = 0, tsdf_prev = 0;

    /** Do NOT use #pragma unroll: it will make it slow **/
    float t_curr = t_min;
    while (t_curr < t_max) {
        Vector3f Xv_t = camera_origin_v + t_curr * ray_v;
        Vector3f X_t = volume_to_voxelf(Xv_t);

        if (!InVolumef(X_t)) return ret;

        float tsdf_curr = this->tsdf(X_t.ToVectori());

        float step_size = tsdf_curr == 0 ?
                          sdf_trunc_ : fmaxf(tsdf_curr, voxel_length_);

        if (tsdf_prev > 0 && tsdf_curr < 0) { /** Zero crossing **/
            float t_intersect = (t_curr * tsdf_prev - t_prev * tsdf_curr)
                / (tsdf_prev - tsdf_curr);

            Vector3f Xv_surface_t = camera_origin_v + t_intersect * ray_v;
            Vector3f X_surface_t = volume_to_voxelf(Xv_surface_t);
            Vector3f normal_v_t = GradientAt(X_surface_t).normalized();
            return transform_camera_to_world.Inverse().Rotate(
                transform_volume_to_world_.Rotate(normal_v_t));
        }

        tsdf_prev = tsdf_curr;
        t_prev = t_curr;
        t_curr += step_size;
    }

    return ret;
}

/**
 * Client end
 */
template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda() {}

template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda(
    float voxel_length, float sdf_trunc,
    TransformCuda &volume_to_world) {

    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    transform_volume_to_world_ = volume_to_world;

    Create();
}

template<size_t N>
UniformTSDFVolumeCuda<N>::UniformTSDFVolumeCuda(
    const UniformTSDFVolumeCuda<N> &other) {

    server_ = other.server();
    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

template<size_t N>
UniformTSDFVolumeCuda<N> &UniformTSDFVolumeCuda<N>::operator=(
    const UniformTSDFVolumeCuda<N> &other) {
    if (this != &other) {
        server_ = other.server();
        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }
    return *this;
}

template<size_t N>
UniformTSDFVolumeCuda<N>::~UniformTSDFVolumeCuda() {
    Release();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Create() {
    if (server_ != nullptr) {
        PrintError("Already created, stop re-creating!\n");
        return;
    }

    server_ = std::make_shared<UniformTSDFVolumeCudaServer<N>>();
    const size_t NNN = N * N * N;
    CheckCuda(cudaMalloc(&(server_->tsdf_), sizeof(float) * NNN));
    CheckCuda(cudaMalloc(&(server_->weight_), sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&(server_->color_), sizeof(Vector3b) * NNN));

    UpdateServer();
    Reset();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->tsdf_));
        CheckCuda(cudaFree(server_->weight_));
        CheckCuda(cudaFree(server_->color_));
    }

    server_ = nullptr;
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->voxel_length_ = voxel_length_;
        server_->inv_voxel_length_ = 1.0f / voxel_length_;

        server_->sdf_trunc_ = sdf_trunc_;
        server_->transform_volume_to_world_ = transform_volume_to_world_;
        server_->transform_world_to_volume_ =
            transform_volume_to_world_.Inverse();
    }
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Reset() {
    if (server_ != nullptr) {
        const size_t NNN = N * N * N;
        CheckCuda(cudaMemset(server_->tsdf_, 0, sizeof(float) * NNN));
        CheckCuda(cudaMemset(server_->weight_, 0, sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(server_->color_, 0, sizeof(Vector3b) * NNN));
    }
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::UploadVolume(std::vector<float> &tsdf,
                                            std::vector<uchar> &weight,
                                            std::vector<Vector3b> &color) {
    if (server_ == nullptr) {
        PrintError("Server not available!\n");
        return;
    }

    const size_t NNN = N * N * N;
    assert(tsdf.size() == NNN);
    assert(weight.size() == NNN);
    assert(color.size() == NNN);

    CheckCuda(cudaMemcpy(server_->tsdf_, tsdf.data(),
                         sizeof(float) * NNN,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->weight_, weight.data(),
                         sizeof(uchar) * NNN,
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(server_->color_, color.data(),
                         sizeof(Vector3b) * NNN,
                         cudaMemcpyHostToDevice));
}

template<size_t N>
std::tuple<std::vector<float>, std::vector<uchar>, std::vector<Vector3b>>
UniformTSDFVolumeCuda<N>::DownloadVolume() {
    std::vector<float> tsdf;
    std::vector<uchar> weight;
    std::vector<Vector3b> color;

    if (server_ == nullptr) {
        PrintError("Server not available!\n");
        return std::make_tuple(tsdf, weight, color);
    }

    const size_t NNN = N * N * N;
    tsdf.resize(NNN);
    weight.resize(NNN);
    color.resize(NNN);

    CheckCuda(cudaMemcpy(tsdf.data(), server_->tsdf_,
                         sizeof(float) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(weight.data(), server_->weight_,
                         sizeof(uchar) * NNN,
                         cudaMemcpyDeviceToHost));
    CheckCuda(cudaMemcpy(color.data(), server_->color_,
                         sizeof(Vector3b) * NNN,
                         cudaMemcpyDeviceToHost));

    return std::make_tuple(
        std::move(tsdf), std::move(weight), std::move(color));
}

/** Reserved for ScalableTSDFVolumeCuda **/
template<size_t N>
__device__
inline void UniformTSDFVolumeCudaServer<N>::Create(
    float *tsdf, uchar *weight, open3d::Vector3b *color) {
    tsdf_ = tsdf;
    weight_ = weight;
    color_ = color;
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Integrate(ImageCuda<open3d::Vector1f> &depth,
                                         MonoPinholeCameraCuda &camera,
                                         TransformCuda &transform_camera_to_world) {
    const int num_blocks = DIV_CEILING(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel << < blocks, threads >> > (
        *server_, *depth.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::RayCasting(ImageCuda<open3d::Vector3f> &image,
                                          MonoPinholeCameraCuda &camera,
                                          TransformCuda &transform_camera_to_world) {
    const dim3 blocks(DIV_CEILING(image.width(), THREAD_2D_UNIT),
                      DIV_CEILING(image.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel << < blocks, threads >> > (
        *server_, *image.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}