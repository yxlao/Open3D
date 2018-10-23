//
// Created by wei on 10/10/18.
//

#pragma once

#include "ScalableTSDFVolumeCuda.h"
#include <Cuda/Container/HashTableCuda.cuh>
#include <Cuda/Container/HashTableCudaKernel.cuh>
#include <Cuda/Container/MemoryHeapCuda.cuh>
#include <Cuda/Container/MemoryHeapCudaKernel.cuh>
#include <Core/Core.h>

namespace open3d {
/** Coordinate system conversions **/
template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::world_to_voxel(
    float xw, float yw, float zw) {
    return world_to_voxel(Vector3f(xw, yw, zw));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::world_to_voxel(
    const Vector3f &Xw) {
    /** Coordinate transform **/
    return volume_to_voxel(transform_world_to_volume_ * Xw);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_to_world(
    float x, float y, float z) {
    return voxel_to_world(Vector3f(x, y, z));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_to_world(
    const Vector3f &X) {
    return transform_volume_to_world_ * voxel_to_volume(X);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_to_volume(
    float x, float y, float z) {
    return Vector3f((x + 0.5f) * voxel_length_,
                    (y + 0.5f) * voxel_length_,
                    (z + 0.5f) * voxel_length_);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_to_volume(
    const Vector3f &X) {
    return voxel_to_volume(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::volume_to_voxel(
    float xv, float yv, float zv) {
    return Vector3f(xv * inv_voxel_length_ - 0.5f,
                    yv * inv_voxel_length_ - 0.5f,
                    zv * inv_voxel_length_ - 0.5f);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::volume_to_voxel(
    const Vector3f &Xv) {
    return volume_to_voxel(Xv(0), Xv(1), Xv(2));
}

/** Voxel coordinate in global volume -> in subvolume **/
template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_locate_subvolume(
    int x, int y, int z) {
    x = x < 0 ? x - (int(N) - 1) : x;
    y = y < 0 ? y - (int(N) - 1) : y;
    z = z < 0 ? z - (int(N) - 1) : z;
    return Vector3i(x / int(N), y / int(N), z / int(N));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_locate_subvolume(
    const Vector3i &X) {
    return voxel_locate_subvolume(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxelf_locate_subvolume(
    float x, float y, float z) {
    return Vector3i(int(floor(x / N)), int(floor(y / N)), int(floor(z / N)));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxelf_locate_subvolume(
    const Vector3f &X) {
    return voxelf_locate_subvolume(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_global_to_local(
    int x, int y, int z, const Vector3i &Xsv) {
    return Vector3i(x - Xsv(0) * int(N),
                    y - Xsv(1) * int(N),
                    z - Xsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_global_to_local(
    const Vector3i &X, const Vector3i &Xsv) {
    return voxel_global_to_local(X(0), X(1), X(2), Xsv);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_global_to_local(
    float x, float y, float z, const Vector3i &Xsv) {
    return Vector3f(x - Xsv(0) * N,
                    y - Xsv(1) * N,
                    z - Xsv(2) * N);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_global_to_local(
    const Vector3f &X, const Vector3i &Xsv) {
    return voxelf_global_to_local(X(0), X(1), X(2), Xsv);
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_local_to_global(
    int xlocal, int ylocal, int zlocal, const Vector3i &Xsv) {
    return Vector3i(xlocal + Xsv(0) * int(N),
                    ylocal + Xsv(1) * int(N),
                    zlocal + Xsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_local_to_global(
    const Vector3i &Xlocal, const Vector3i &Xsv) {
    return voxel_local_to_global(Xlocal(0), Xlocal(1), Xlocal(2), Xsv);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_local_to_global(
    float xlocal, float ylocal, float zlocal, const Vector3i &Xsv) {
    return Vector3f(xlocal + Xsv(0) * N,
                    ylocal + Xsv(1) * N,
                    zlocal + Xsv(2) * N);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_local_to_global(
    const Vector3f &Xlocal, const Vector3i &Xsv) {
    return voxelf_local_to_global(Xlocal(0), Xlocal(1), Xlocal(2), Xsv);
}

template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundary(
    int xlocal, int ylocal, int zlocal, bool for_gradient) {
    return !for_gradient
        || (xlocal == 0 || xlocal == N - 1
            || ylocal == 0 || ylocal == N - 1
            || zlocal == 0 || zlocal == N - 1);
}

template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundary(
    const Vector3i &Xlocal, bool for_gradient) {
    return OnBoundary(Xlocal(0), Xlocal(1), Xlocal(2), for_gradient);
}

template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundaryf(
    float xlocal, float ylocal, float zlocal, bool for_gradient) {
    return (xlocal >= N - 1 || ylocal >= N - 1 || zlocal >= N - 1)
        && (!for_gradient
            || (xlocal < 1 || xlocal >= N - 2
                || ylocal < 1 || ylocal >= N - 2
                || zlocal < 1 || zlocal >= N - 2));
}

template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundaryf(
    const Vector3f &Xlocal, bool for_gradient) {
    return OnBoundaryf(Xlocal(0), Xlocal(1), Xlocal(2), for_gradient);
}

/** Query **/
template<size_t N>
__device__
UniformTSDFVolumeCudaServer<N> *
ScalableTSDFVolumeCudaServer<N>::QuerySubvolume(
    const Vector3i &Xsv) {
    return hash_table_[Xsv];
}

template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::QuerySubvolumeWithNeighborIndex(
    const Vector3i &Xsv, int dxsv, int dysv, int dzsv,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {
    subvolumes[LinearizeNeighborIndex(dxsv, dysv, dzsv)]
        = hash_table_[Vector3i(Xsv(0) + dxsv, Xsv(1) + dysv, Xsv(2) + dzsv)];
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::NeighborIndexOfBoundaryVoxel(
    int xlocal, int ylocal, int zlocal) {
    return Vector3i(xlocal < 0 ? -1 : (xlocal >= N ? 1 : 0),
                    ylocal < 0 ? -1 : (ylocal >= N ? 1 : 0),
                    zlocal < 0 ? -1 : (zlocal >= N ? 1 : 0));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::NeighborIndexOfBoundaryVoxel(
    const Vector3i &Xlocal) {
    return NeighborIndexOfBoundaryVoxel(Xlocal(0), Xlocal(1), Xlocal(2));
}

template<size_t N>
__device__
inline int ScalableTSDFVolumeCudaServer<N>::LinearizeNeighborIndex(
    int dxsv, int dysv, int dzsv) {
    //return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
    return 9 * dzsv + 3 * dysv + dxsv + 13;
}

template<size_t N>
__device__
inline int ScalableTSDFVolumeCudaServer<N>::LinearizeNeighborIndex(
    const Vector3i &dXsv) {
    return LinearizeNeighborIndex(dXsv(0), dXsv(1), dXsv(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    int xlocal, int ylocal, int zlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(0 <= xlocal && xlocal < N);
    assert(0 <= ylocal && ylocal < N);
    assert(0 <= zlocal && zlocal < N);
#endif
    Vector3f n = Vector3f::Zeros();
    Vector3i X = Vector3i(xlocal, ylocal, zlocal);

    Vector3i X0 = X, X1 = X;

#pragma unroll 1
    for (size_t i = 0; i < 3; ++i) {
        X0(i) -= 1;
        X1(i) += 1;

        Vector3i dXsv0 = NeighborIndexOfBoundaryVoxel(X0);
        Vector3i dXsv1 = NeighborIndexOfBoundaryVoxel(X1);

        UniformTSDFVolumeCudaServer<N> *subvolume0 =
            subvolumes[LinearizeNeighborIndex(dXsv0)];
        UniformTSDFVolumeCudaServer<N> *subvolume1 =
            subvolumes[LinearizeNeighborIndex(dXsv1)];
        float tsdf0 = (subvolume0 == nullptr) ? 0 :
                      subvolume0->tsdf(X0 - dXsv0 * N);
        float tsdf1 = (subvolume1 == nullptr) ? 0 :
                      subvolume1->tsdf(X1 - dXsv1 * N);
        n(i) = tsdf1 - tsdf0;

        X0(i) = X(i);
        X1(i) = X(i);
    }

    return n;
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    const Vector3i &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes) {
    return gradient(Xlocal(0), Xlocal(1), Xlocal(2), subvolumes);
}

template<size_t N>
__device__
float ScalableTSDFVolumeCudaServer<N>::TSDFOnBoundaryAt(
    float x, float y, float z,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= x && x < N + 1);
    assert(-1 <= y && y < N + 1);
    assert(-1 <= z && z < N + 1);
#endif

    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Vector3f(x - floorf(x), y - floorf(y), z - floorf(z));
    Vector3f rneg = Vector3f(1 - r(0), 1 - r(1), 1 - r(2));

    float sum_weight_interp = 0;
    float sum_tsdf = 0;
    for (int i = 0; i < 8; ++i) {
        Vector3i dX_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + dX_i;

        Vector3i dXsv_i = NeighborIndexOfBoundaryVoxel(dX_i);

        UniformTSDFVolumeCudaServer<N> *subvolume =
            subvolumes[LinearizeNeighborIndex(dXsv_i)];

        float tsdf_i = (subvolume == nullptr) ? 0.0f :
                       subvolume->tsdf(X_i(0) - N * dXsv_i(0),
                                       X_i(1) - N * dXsv_i(1),
                                       X_i(2) - N * dXsv_i(2));
        float weight_interp_i = (subvolume == nullptr) ? 0.0f :
                                (rneg(0) * (1 - dX_i(0)) + r(0) * dX_i(0)) *
                                    (rneg(1) * (1 - dX_i(1)) + r(1) * dX_i(1)) *
                                    (rneg(2) * (1 - dX_i(2)) + r(2) * dX_i(2));

        sum_tsdf += weight_interp_i * tsdf_i;
        sum_weight_interp += weight_interp_i;
    }

    return sum_weight_interp > 0 ? sum_tsdf / sum_weight_interp : 0;
}

template<size_t N>
__device__
float ScalableTSDFVolumeCudaServer<N>::TSDFOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {
    return TSDFOnBoundaryAt(Xlocal(0), Xlocal(1), Xlocal(2), subvolumes);
}

template<size_t N>
__device__
uchar ScalableTSDFVolumeCudaServer<N>::WeightOnBoundaryAt(
    float xlocal, float ylocal, float zlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= xlocal && xlocal < N + 1);
    assert(-1 <= ylocal && ylocal < N + 1);
    assert(-1 <= zlocal && zlocal < N + 1);
#endif

    Vector3i X = Vector3i(int(xlocal), int(ylocal), int(zlocal));
    Vector3f r = Vector3f(xlocal - floorf(xlocal),
                          ylocal - floorf(ylocal),
                          zlocal - floorf(zlocal));
    Vector3f rneg = Vector3f(1 - r(0), 1 - r(1), 1 - r(2));

    float sum_weight_interp = 0;
    float sum_weight = 0;
    for (int i = 0; i < 8; ++i) {
        Vector3i dX_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + dX_i;

        Vector3i dXsv_i = NeighborIndexOfBoundaryVoxel(dX_i);

        UniformTSDFVolumeCudaServer<N> *subvolume =
            subvolumes[LinearizeNeighborIndex(dXsv_i)];

        float weight_i = (subvolume == nullptr) ? 0.0f :
                         subvolume->weight(X_i(0) - N * dXsv_i(0),
                                           X_i(1) - N * dXsv_i(1),
                                           X_i(2) - N * dXsv_i(2));
        float weight_interp_i = (subvolume == nullptr) ? 0.0f :
                                (rneg(0) * (1 - dX_i(0)) + r(0) * dX_i(0)) *
                                    (rneg(1) * (1 - dX_i(1)) + r(1) * dX_i(1)) *
                                    (rneg(2) * (1 - dX_i(2)) + r(2) * dX_i(2));

        sum_weight += weight_interp_i * weight_i;
        sum_weight_interp += weight_interp_i;
    }

    return sum_weight_interp > 0 ?
           uchar(fminf(sum_weight / sum_weight_interp, 255)) : uchar(0);
}

template<size_t N>
__device__
uchar ScalableTSDFVolumeCudaServer<N>::WeightOnBoundaryAt(
    const Vector3f &Xlocal, UniformTSDFVolumeCudaServer<N> **subvolumes) {
    return WeightOnBoundaryAt(Xlocal(0), Xlocal(1), Xlocal(2), subvolumes);
}

template<size_t N>
__device__
Vector3b ScalableTSDFVolumeCudaServer<N>::ColorOnBoundaryAt(
    float xlocal, float ylocal, float zlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= xlocal && xlocal < N + 1);
    assert(-1 <= ylocal && ylocal < N + 1);
    assert(-1 <= zlocal && zlocal < N + 1);
#endif

    Vector3i X = Vector3i(int(xlocal), int(ylocal), int(zlocal));
    Vector3f r = Vector3f(xlocal - floorf(xlocal),
                          ylocal - floorf(ylocal),
                          zlocal - floorf(zlocal));
    Vector3f rneg = Vector3f(1 - r(0), 1 - r(1), 1 - r(2));

    float sum_weight_interp = 0;
    Vector3f sum_color = Vector3f::Zeros();
    for (int i = 0; i < 8; ++i) {
        Vector3i dX_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + dX_i;

        Vector3i dXsv_i = NeighborIndexOfBoundaryVoxel(dX_i);

        UniformTSDFVolumeCudaServer<N> *subvolume =
            subvolumes[LinearizeNeighborIndex(dXsv_i)];

        Vector3f color_i = (subvolume == nullptr) ? Vector3f(0) :
                           subvolume->color(X_i(0) - N * dXsv_i(0),
                                            X_i(1) - N * dXsv_i(1),
                                            X_i(2) - N * dXsv_i(2)).ToVectorf();
        float weight_interp_i = (subvolume == nullptr) ? 0.0f :
                                (rneg(0) * (1 - dX_i(0)) + r(0) * dX_i(0)) *
                                    (rneg(1) * (1 - dX_i(1)) + r(1) * dX_i(1)) *
                                    (rneg(2) * (1 - dX_i(2)) + r(2) * dX_i(2));

        sum_color += weight_interp_i * color_i;
        sum_weight_interp += weight_interp_i;
    }

    return sum_weight_interp > 0 ?
           Vector3b(uchar(sum_color(0) / sum_weight_interp),
                    uchar(sum_color(1) / sum_weight_interp),
                    uchar(sum_color(2) / sum_weight_interp)) : Vector3b(0);
}

template<size_t N>
__device__
Vector3b ScalableTSDFVolumeCudaServer<N>::ColorOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {
    return ColorOnBoundaryAt(Xlocal(0), Xlocal(1), Xlocal(2), subvolumes);
}

template<size_t N>
__device__
Vector3f ScalableTSDFVolumeCudaServer<N>::GradientOnBoundaryAt(
    float xlocal, float ylocal, float zlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {

    Vector3f n = Vector3f::Zeros();
    Vector3f X = Vector3f(xlocal, ylocal, zlocal);
    Vector3f X0 = X, X1 = X;

    const float half_gap = voxel_length_;
#pragma unroll 1
    for (size_t i = 0; i < 3; ++i) {
        X0(i) -= half_gap;
        X1(i) += half_gap;
        n(i) = TSDFOnBoundaryAt(X1, subvolumes)
            - TSDFOnBoundaryAt(X0, subvolumes);

        X0(i) = X(i);
        X1(i) = X(i);
    }
    return n;
}

template<size_t N>
__device__
Vector3f ScalableTSDFVolumeCudaServer<N>::GradientOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **subvolumes) {
    return GradientOnBoundaryAt(Xlocal(0), Xlocal(1), Xlocal(2), subvolumes);
}

/** High level functions **/
template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::TouchSubvolume(
    int x, int y,
    ImageCudaServer<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    float d = depth.get(x, y)(0);

    /** TODO: wrap the criteria in depth image (RGBD-Image) **/
    if (d < 0.1f || d > 3.0f) return;

    Vector3f Xw_near = transform_camera_to_world * camera.InverseProjection(
        x, y, d - sdf_trunc_);
    Vector3i Xsv_near = voxel_locate_subvolume(world_to_voxel(Xw_near));

    Vector3f Xw_far = transform_camera_to_world * camera.InverseProjection(
        x, y, d + sdf_trunc_);
    Vector3i Xsv_far = voxel_locate_subvolume(world_to_voxel(Xw_far));

    /** 3D line from Xsv_near to Xsv_far **/
    /** https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm) **/
    Vector3i DXsv = Xsv_far - Xsv_near;
    Vector3i DXsv_abs = Vector3i(abs(DXsv(0)), abs(DXsv(1)), abs(DXsv(2)));
    int step = DXsv_abs(0) >= DXsv_abs(1) ? DXsv_abs(0) : DXsv_abs(1);
    step = DXsv_abs(2) >= step ? DXsv_abs(2) : step;
    Vector3f DXsv_normalized = DXsv.ToVectorf() * (1.0f / step);

    Vector3f Xsv_curr = Xsv_near.ToVectorf();

    HashEntry<Vector3i> entry;
    for (int i = 0; i <= step; ++i) {
        int ptr = hash_table_.New(Vector3i(
            int(Xsv_curr(0)), int(Xsv_curr(1)), int(Xsv_curr(2))));

        if (ptr >= 0) {
            entry.key = Xsv_curr;
            entry.value_ptr = ptr;
            target_subvolume_entry_array_.push_back(entry);
        }

        Xsv_curr += DXsv_normalized;
    }
}

/**
 * Client end
 */
template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda() {
    bucket_count_ = -1;
    value_capacity_ = -1;
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda(
    int bucket_count, int value_capacity,
    float voxel_length, float sdf_trunc,
    TransformCuda &transform_volume_to_world) {

    bucket_count_ = bucket_count;
    value_capacity_ = value_capacity;

    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    transform_volume_to_world_ = transform_volume_to_world;

    Create(bucket_count, value_capacity);
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::ScalableTSDFVolumeCuda(
    const ScalableTSDFVolumeCuda<N> &other) {
    server_ = other.server();
    hash_table_ = other.hash_table();

    bucket_count_ = other.bucket_count_;
    value_capacity_ = other.value_capacity_;

    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

template<size_t N>
ScalableTSDFVolumeCuda<N> &ScalableTSDFVolumeCuda<N>::operator=(
    const ScalableTSDFVolumeCuda<N> &other) {
    if (this != &other) {
        Release();

        server_ = other.server();
        hash_table_ = other.hash_table();

        bucket_count_ = other.bucket_count_;
        value_capacity_ = other.value_capacity_;

        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }

    return *this;
}

template<size_t N>
ScalableTSDFVolumeCuda<N>::~ScalableTSDFVolumeCuda() {
    Release();
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Create(
    int bucket_count, int value_capacity) {
    if (server_ != nullptr) {
        PrintError("Already created, stop re-creating!\n");
        return;
    }

    server_ = std::make_shared<ScalableTSDFVolumeCudaServer<N>>();
    hash_table_.Create(bucket_count, value_capacity);
    target_subvolume_entry_array_.Create(value_capacity);

    /** Comparing to 512^3, we can have at most (512^2) 8^3 cubes.
     * That is 262144. **/
    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->tsdf_memory_pool_,
                         sizeof(float) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->weight_memory_pool_,
                         sizeof(uchar) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->color_memory_pool_,
                         sizeof(Vector3b) * NNN * value_capacity));

    UpdateServer();

    const dim3 threads(THREAD_1D_UNIT);
    const dim3 blocks(DIV_CEILING(value_capacity, THREAD_1D_UNIT));
    CreateScalableTSDFVolumesKernel << < blocks, threads >> > (*server_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Release() {
    if (server_ != nullptr && server_.use_count() == 1) {
        CheckCuda(cudaFree(server_->tsdf_memory_pool_));
        CheckCuda(cudaFree(server_->weight_memory_pool_));
        CheckCuda(cudaFree(server_->color_memory_pool_));
    }

    server_ = nullptr;
    hash_table_.Release();
    target_subvolume_entry_array_.Release();
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->hash_table_ = *hash_table_.server();
        server_->target_subvolume_entry_array_ =
            *target_subvolume_entry_array_.server();

        server_->voxel_length_ = voxel_length_;
        server_->inv_voxel_length_ = 1.0f / voxel_length_;
        server_->sdf_trunc_ = sdf_trunc_;
        server_->transform_volume_to_world_ = transform_volume_to_world_;
        server_->transform_world_to_volume_ =
            transform_volume_to_world_.Inverse();
    }
}

template<size_t N>
std::pair<std::vector<Vector3i>,
          std::vector<std::tuple<std::vector<float>,
                                 std::vector<uchar>,
                                 std::vector<Vector3b>>>>
ScalableTSDFVolumeCuda<N>::DownloadVolumes() {

    auto hash_table = hash_table_.Download();
    std::vector<Vector3i> &keys = std::get<0>(hash_table);
    std::vector<UniformTSDFVolumeCudaServer<N>>
        &volume_servers = std::get<1>(hash_table);

    assert(keys.size() == volume_servers.size());

    std::vector<std::tuple<std::vector<float>,
                           std::vector<uchar>,
                           std::vector<Vector3b>>> volumes;
    volumes.resize(volume_servers.size());

    for (int i = 0; i < volumes.size(); ++i) {
        std::vector<float> tsdf;
        std::vector<uchar> weight;
        std::vector<Vector3b> color;

        const size_t NNN = N * N * N;
        tsdf.resize(NNN);
        weight.resize(NNN);
        color.resize(NNN);

        CheckCuda(cudaMemcpy(tsdf.data(), volume_servers[i].tsdf_,
                             sizeof(float) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(weight.data(), volume_servers[i].weight_,
                             sizeof(uchar) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(color.data(), volume_servers[i].color_,
                             sizeof(Vector3b) * NNN,
                             cudaMemcpyDeviceToHost));

        volumes[i] = std::make_tuple(
            std::move(tsdf), std::move(weight), std::move(color));
    }

    return std::make_pair(std::move(keys), std::move(volumes));
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::TouchBlocks(
    ImageCuda<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const dim3 blocks(DIV_CEILING(depth.width(), THREAD_2D_UNIT),
                      DIV_CEILING(depth.height(), THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    TouchSubvolumesKernel << < blocks, threads >> > (
        *server_, *depth.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::GetSubvolumesInFrustum(
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const dim3 blocks(bucket_count_);
    const dim3 threads(THREAD_1D_UNIT);
    GetSubvolumesInFrustumKernel << < blocks, threads >> > (
        *server_, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Integrate(
    ImageCuda<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const int num_blocks = 0;
    const dim3 blocks(num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateKernel << < blocks, threads >> > (
        *server_, *depth.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::RayCasting(
    ImageCuda<Vector1f> &image,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

}

}