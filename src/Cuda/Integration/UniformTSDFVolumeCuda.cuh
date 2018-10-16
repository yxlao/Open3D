//
// Created by wei on 10/9/18.
//

#pragma once

#include "UniformTSDFVolumeCuda.h"
#include <Core/Core.h>
#include <Cuda/Common/UtilsCuda.h>

namespace open3d {

/**
 * Server end
 */
/** Coordinate conversions **/
template<size_t N>
__device__
inline Vector3i UniformTSDFVolumeCudaServer<N>::Vectorize(size_t index) {
    Vector3i ret;
    ret(0) = int(index % N);
    ret(1) = int((index % (N * N)) / N);
    ret(2) = int(index / (N * N));
    return ret;
}

template<size_t N>
__device__
inline int UniformTSDFVolumeCudaServer<N>::IndexOf(int x, int y, int z) {
    return int(z * (N * N) + y * N + x);
}

template<size_t N>
__device__
inline int UniformTSDFVolumeCudaServer<N>::IndexOf(const Vector3i &X) {
    return IndexOf(X(0), X(1), X(2));
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
inline float &UniformTSDFVolumeCudaServer<N>::tsdf(int x, int y, int z) {
    return tsdf_[IndexOf(x, y, z)];
}
template<size_t N>
__device__
inline float &UniformTSDFVolumeCudaServer<N>::tsdf(const Vector3i &X) {
    return tsdf_[IndexOf(X)];
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::gradient(int x, int y, int z) {
    return Vector3f(tsdf_[IndexOf(min(x + 1, int(N - 1)), y, z)]
                        - tsdf_[IndexOf(max(x - 1, 0), y, z)],
                    tsdf_[IndexOf(x, min(y + 1, int(N - 1)), z)]
                        - tsdf_[IndexOf(x, max(y - 1, 0), z)],
                    tsdf_[IndexOf(x, y, min(z + 1, int(N - 1)))]
                        - tsdf_[IndexOf(x, y, max(z - 1, 0))]);
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::gradient(const Vector3i &X) {
    return gradient(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline uchar &UniformTSDFVolumeCudaServer<N>::weight(int x, int y, int z) {
    return weight_[IndexOf(x, y, z)];
}
template<size_t N>
__device__
inline uchar &UniformTSDFVolumeCudaServer<N>::weight(const Vector3i &X) {
    return weight_[IndexOf(X)];
}

template<size_t N>
__device__
inline Vector3b &UniformTSDFVolumeCudaServer<N>::color(int x, int y, int z) {
    return color_[IndexOf(x, y, z)];
}
template<size_t N>
__device__
inline Vector3b &UniformTSDFVolumeCudaServer<N>::color(const Vector3i &X) {
    return color_[IndexOf(X)];
}

template<size_t N>
__device__
inline Vector3i &UniformTSDFVolumeCudaServer<N>::vertex_indices(
    int x, int y, int z) {
    return vertex_indices_[IndexOf(x, y, z)];
}
template<size_t N>
__device__
inline Vector3i &UniformTSDFVolumeCudaServer<N>::vertex_indices(
    const Vector3i &X) {
    return vertex_indices_[IndexOf(X)];
}

template<size_t N>
__device__
inline int &UniformTSDFVolumeCudaServer<N>::table_index(
    int x, int y, int z) {
    return table_index_[IndexOf(x, y, z)];
}
template<size_t N>
__device__
inline int &UniformTSDFVolumeCudaServer<N>::table_index(
    const Vector3i &X) {
    return table_index_[IndexOf(X)];
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

/** Default invalid value is 0. TODO: Check it later **/
template<size_t N>
__device__
inline float UniformTSDFVolumeCudaServer<N>::TSDFAt(float x, float y, float z) {
    /** If it is in Volume, then all the nearby components are in volume,
     * no boundary check is required **/
    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    assert(X(0) < N - 1);
    assert(X(1) < N - 1);
    assert(X(2) < N - 1);
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
inline float UniformTSDFVolumeCudaServer<N>::TSDFAt(const Vector3f &X) {
    return TSDFAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline uchar UniformTSDFVolumeCudaServer<N>::WeightAt(
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
inline uchar UniformTSDFVolumeCudaServer<N>::WeightAt(const Vector3f &X) {
    return WeightAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3b UniformTSDFVolumeCudaServer<N>::ColorAt(
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
inline Vector3b UniformTSDFVolumeCudaServer<N>::ColorAt(const Vector3f &X) {
    return ColorAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::GradientAt(
    float x, float y, float z) {
    return GradientAt(Vector3f(x, y, z));
}

template<size_t N>
__device__
inline Vector3f UniformTSDFVolumeCudaServer<N>::GradientAt(const Vector3f &X) {
    Vector3f n = Vector3f::Zeros();

    const float half_gap = 0.5f * voxel_length_;
    Vector3f X0 = X, X1 = X;

#pragma unroll 1
    for (size_t i = 0; i < 3; i++) {
        X0(i) = fmaxf(X0(i) - half_gap, 0.01f);
        X1(i) = fminf(X1(i) + half_gap, N - 1 - 0.01f);
        n(i) = (TSDFAt(X1) - TSDFAt(X0)) * inv_voxel_length_;

        X0(i) = X(i);
        X1(i) = X(i);
    }
    return n;
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
    mesh_ = other.mesh();
    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

template<size_t N>
UniformTSDFVolumeCuda<N> &UniformTSDFVolumeCuda<N>::operator=(
    const UniformTSDFVolumeCuda<N> &other) {
    if (this != &other) {
        server_ = other.server();
        mesh_ = other.mesh();
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

    CheckCuda(cudaMalloc(&(server_->vertex_indices_), sizeof(Vector3i) * NNN));

    CheckCuda(cudaMalloc(&(server_->table_index_), sizeof(int) * NNN));

    mesh_.Create(64 * N * N, 64 * N * N);
    UpdateServer();
    Reset();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->tsdf_));
        CheckCuda(cudaFree(server_->weight_));
        CheckCuda(cudaFree(server_->color_));

        CheckCuda(cudaFree(server_->vertex_indices_));
        CheckCuda(cudaFree(server_->table_index_));
        mesh_.Release();
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
    server_->mesh_ = *mesh_.server();
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::Reset() {
    if (server_ != nullptr) {
        const int NNN = N * N * N;
        CheckCuda(cudaMemset(server_->tsdf_, 0, sizeof(float) * NNN));
        CheckCuda(cudaMemset(server_->weight_, 0, sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(server_->color_, 0, sizeof(Vector3b) * NNN));
        CheckCuda(cudaMemset(server_->vertex_indices_, 0xff,
                             sizeof(Vector3i) * NNN));
        CheckCuda(cudaMemset(server_->table_index_, 0, sizeof(int) * NNN));

        mesh_.Reset();
    }
}

template<size_t N>
void UniformTSDFVolumeCuda<N>::UploadVolume(std::vector<float> &tsdf,
                                            std::vector<uchar> &weight,
                                            std::vector<open3d::Vector3b> &color) {
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
int UniformTSDFVolumeCuda<N>::Integrate(ImageCuda<open3d::Vector1f> &depth,
                                        MonoPinholeCameraCuda &camera,
                                        TransformCuda &transform_camera_to_world) {
    const int num_blocks = UPPER_ALIGN(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel << < blocks, threads >> > (
        *server_, *depth.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    return 0;
}

template<size_t N>
int UniformTSDFVolumeCuda<N>::RayCasting(ImageCuda<open3d::Vector3f> &image,
                                         MonoPinholeCameraCuda &camera,
                                         TransformCuda &transform_camera_to_world) {
    const dim3 blocks(UPPER_ALIGN(image.width(), THREAD_2D_UNIT),
                      UPPER_ALIGN(image.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel << < blocks, threads >> > (
        *server_, *image.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    return 0;
}

template<size_t N>
int UniformTSDFVolumeCuda<N>::MarchingCubes() {
    const int num_blocks = UPPER_ALIGN(N, THREAD_3D_UNIT);
    const dim3 blocks(num_blocks, num_blocks, num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);

    Timer timer;
    timer.Start();
    MarchingCubesVertexAllocationKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration());

    timer.Start();
    MarchingCubesVertexExtractionKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration());

    timer.Start();
    MarchingCubesTriangleExtractionKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
    timer.Stop();
    PrintInfo("Triangulation takes %f milliseconds\n", timer.GetDuration());
    return 0;
}
}