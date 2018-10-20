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
inline bool UniformTSDFVolumeCudaServer<N>::InVolume(int x, int y, int z) {
    return 0 <= x && x < N && 0 <= y && y < N && 0 <= z && z < N;
}
template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaServer<N>::InVolume(const Vector3i &X) {
    return InVolume(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaServer<N>::InVolumef(
    float x, float y, float z) {
    return 0 <= x && x < (N - 1)
        && 0 <= y && y < (N - 1)
        && 0 <= z && z < (N - 1);
}

template<size_t N>
__device__
inline bool UniformTSDFVolumeCudaServer<N>::InVolumef(const Vector3f &X) {
    return InVolumef(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::world_to_voxel(
    float x, float y, float z) {
    return world_to_voxel(Vector3f(x, y, z));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::world_to_voxel(
    const Vector3f &X) {
    /** Coordinate transform **/
    return volume_to_voxel(transform_world_to_volume_ * X);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxel_to_world(
    float x, float y, float z) {
    return voxel_to_world(Vector3f(x, y, z));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxel_to_world(
    const Vector3f &X_v) {
    return transform_volume_to_world_ * voxel_to_volume(X_v);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxel_to_volume(
    float x, float y, float z) {
    return Vector3f((x + 0.5f) * voxel_length_, /** Scale transform **/
                    (y + 0.5f) * voxel_length_,
                    (z + 0.5f) * voxel_length_);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::voxel_to_volume(
    const Vector3f &X) {
    return voxel_to_volume(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::volume_to_voxel(
    float x, float y, float z) {
    return Vector3f(x * inv_voxel_length_ - 0.5f,
                    y * inv_voxel_length_ - 0.5f,
                    z * inv_voxel_length_ - 0.5f);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::volume_to_voxel(
    const Vector3f &X) {
    return volume_to_voxel(X(0), X(1), X(2));
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::gradient(int x, int y, int z) {
    return Vector3f(
        tsdf_[IndexOf(min(x + 1, int(N - 1)), y, z)]
            - tsdf_[IndexOf(max(x - 1, 0), y, z)],
        tsdf_[IndexOf(x, min(y + 1, int(N - 1)), z)]
            - tsdf_[IndexOf(x, max(y - 1, 0), z)],
        tsdf_[IndexOf(x, y, min(z + 1, int(N - 1)))]
            - tsdf_[IndexOf(x, y, max(z - 1, 0))]);
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::gradient(const Vector3i &X) {
    return gradient(X(0), X(1), X(2));
}

/** Interpolations. TODO: Check default value later **/
template<size_t N>
__device__
float UniformTSDFVolumeCudaServer<N>::TSDFAt(float x, float y, float z) {
    /** If it is in Volume, then all the nearby components are in volume,
     * no boundary check is required **/
    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    return (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf_[IndexOf(X + Vector3i(0, 0, 0))] +
                r(2) * tsdf_[IndexOf(X + Vector3i(0, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * tsdf_[IndexOf(X + Vector3i(0, 1, 0))] +
                r(2) * tsdf_[IndexOf(X + Vector3i(0, 1, 1))]
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf_[IndexOf(X + Vector3i(1, 0, 0))] +
                r(2) * tsdf_[IndexOf(X + Vector3i(1, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * tsdf_[IndexOf(X + Vector3i(1, 1, 0))] +
                r(2) * tsdf_[IndexOf(X + Vector3i(1, 1, 1))]
        ));
}

template<size_t N>
__device__
float UniformTSDFVolumeCudaServer<N>::TSDFAt(const Vector3f &X) {
    return TSDFAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
uchar UniformTSDFVolumeCudaServer<N>::WeightAt(
    float x, float y, float z) {
    Vector3f Xf = Vector3f(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    return uchar((1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight_[IndexOf(X + Vector3i(0, 0, 0))] +
                r(2) * weight_[IndexOf(X + Vector3i(0, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * weight_[IndexOf(X + Vector3i(0, 1, 0))] +
                r(2) * weight_[IndexOf(X + Vector3i(0, 1, 1))]
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight_[IndexOf(X + Vector3i(1, 0, 0))] +
                r(2) * weight_[IndexOf(X + Vector3i(1, 0, 1))]
        ) + r(1) * (
            (1 - r(2)) * weight_[IndexOf(X + Vector3i(1, 1, 0))] +
                r(2) * weight_[IndexOf(X + Vector3i(1, 1, 1))]
        )));
}

template<size_t N>
__device__
uchar UniformTSDFVolumeCudaServer<N>::WeightAt(const Vector3f &X) {
    return WeightAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
Vector3b UniformTSDFVolumeCudaServer<N>::ColorAt(
    float x, float y, float z) {

    Vector3f Xf = Vector3f(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    Vector3f colorf = (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(X + Vector3i(0, 0, 0))].ToVectorf() +
                r(2) * color_[IndexOf(X + Vector3i(0, 0, 1))].ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(X + Vector3i(0, 1, 0))].ToVectorf() +
                r(2) * color_[IndexOf(X + Vector3i(0, 1, 1))].ToVectorf()
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * color_[IndexOf(X + Vector3i(1, 0, 0))].ToVectorf() +
                r(2) * color_[IndexOf(X + Vector3i(1, 0, 1))].ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color_[IndexOf(X + Vector3i(1, 1, 0))].ToVectorf() +
                r(2) * color_[IndexOf(X + Vector3i(1, 1, 1))].ToVectorf()
        ));

    return Vector3b(uchar(colorf(0)), uchar(colorf(1)), uchar(colorf(2)));
}

template<size_t N>
__device__
Vector3b UniformTSDFVolumeCudaServer<N>::ColorAt(const Vector3f &X) {
    return ColorAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::GradientAt(
    float x, float y, float z) {
    return GradientAt(Vector3f(x, y, z));
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::GradientAt(const Vector3f &X) {
    Vector3f n = Vector3f::Zeros();

    const float half_gap = 0.5f * voxel_length_;
    const float epsilon = 0.1f * voxel_length_;
    Vector3f X0 = X, X1 = X;

#pragma unroll 1
    for (size_t i = 0; i < 3; i++) {
        X0(i) = fmaxf(X0(i) - half_gap, epsilon);
        X1(i) = fminf(X1(i) + half_gap, N - 1 - epsilon);
        n(i) = (TSDFAt(X1) - TSDFAt(X0)) * inv_voxel_length_;

        X0(i) = X(i);
        X1(i) = X(i);
    }
    return n;
}

/** High level methods **/
template<size_t N>
__device__
void UniformTSDFVolumeCudaServer<N>::Integrate(
    int x, int y, int z,
    ImageCudaServer<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    /** Projective data association **/
    Vector3f X_w = voxel_to_world(x, y, z);
    Vector3f X_c = transform_camera_to_world.Inverse() * X_w;
    Vector2f p = camera.Projection(X_c);

    /** TSDF **/
    if (!camera.IsValid(p)) return;
    float d = depth.get_interp(p(0), p(1))(0);

    float sdf = d - X_c(2);
    if (sdf <= - sdf_trunc_) return;
    sdf = fminf(sdf, sdf_trunc_);

    /** Weight average **/
    /** TODO: color **/
    float &tsdf = this->tsdf(x, y, z);
    uchar &weight = this->weight(x, y, z);
    tsdf = (tsdf * weight + sdf * 1.0f) / (weight + 1.0f);
    weight = uchar(fminf(weight + 1.0f, 255));
}

template<size_t N>
__device__
Vector3f UniformTSDFVolumeCudaServer<N>::RayCasting(
    int x, int y,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    Vector3f ret = Vector3f(0);

    Vector3f ray_c = camera.InverseProjection(x, y, 1.0f).normalized();
    /** TODO: throw it into parameters **/
    const float t_min = 0.2f / ray_c(2);
    const float t_max = 3.0f / ray_c(2);

    const Vector3f camera_origin_v = transform_world_to_volume_ *
        (transform_camera_to_world * Vector3f(0));
    const Vector3f ray_v = transform_world_to_volume_.Rotate(
        transform_camera_to_world.Rotate(ray_c));

    float t_prev = 0, tsdf_prev = 0;

    /** Do NOT use #pragma unroll: it will make it slow **/
    float t_curr = t_min;
    while (t_curr < t_max) {
        Vector3f X_v = camera_origin_v + t_curr * ray_v;
        Vector3f X_voxel = volume_to_voxel(X_v);

        if (!InVolumef(X_voxel)) return ret;

        float tsdf_curr = this->tsdf(X_voxel);

        float step_size = tsdf_curr == 0 ?
                          sdf_trunc_ : fmaxf(tsdf_curr, voxel_length_);

        if (tsdf_prev > 0 && tsdf_curr < 0) { /** Zero crossing **/
            float t_intersect = (t_curr * tsdf_prev - t_prev * tsdf_curr)
                / (tsdf_prev - tsdf_curr);

            Vector3f X_surface_v = camera_origin_v + t_intersect * ray_v;
            Vector3f X_surface_voxel = volume_to_voxel(X_surface_v);
            Vector3f normal_v = GradientAt(X_surface_voxel).normalized();

            return transform_camera_to_world.Inverse().Rotate(
                transform_volume_to_world_.Rotate(normal_v));
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
    server_->voxel_length_ = voxel_length_;
    server_->inv_voxel_length_ = 1.0f / voxel_length_;

    server_->sdf_trunc_ = sdf_trunc_;
    server_->transform_volume_to_world_ = transform_volume_to_world_;
    server_->transform_world_to_volume_ = transform_volume_to_world_.Inverse();
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