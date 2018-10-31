//
// Created by wei on 10/10/18.
//

#pragma once

#include "ScalableTSDFVolumeCuda.h"
#include "MarchingCubesConstCuda.h"
#include <Cuda/Container/HashTableCuda.cuh>
#include <Cuda/Container/HashTableCudaKernel.cuh>
#include <Cuda/Container/MemoryHeapCuda.cuh>
#include <Cuda/Container/MemoryHeapCudaKernel.cuh>
#include <Core/Core.h>

namespace open3d {

/** Coordinate system conversions **/
template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::world_to_voxelf(
    const Vector3f &Xw) {
    return volume_to_voxelf(transform_world_to_volume_ * Xw);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_to_world(
    const Vector3f &X) {
    return transform_volume_to_world_ * voxelf_to_volume(X);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_to_volume(
    const Vector3f &X) {
    return Vector3f((X(0) + 0.5f) * voxel_length_,
                    (X(1) + 0.5f) * voxel_length_,
                    (X(2) + 0.5f) * voxel_length_);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::volume_to_voxelf(
    const Vector3f &Xv) {
    return Vector3f(Xv(0) * inv_voxel_length_ - 0.5f,
                    Xv(1) * inv_voxel_length_ - 0.5f,
                    Xv(2) * inv_voxel_length_ - 0.5f);
}

/** Voxel coordinate in global volume -> in subvolume **/
template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_locate_subvolume(
    const Vector3i &X) {
    return Vector3i((X(0) < 0 ? X(0) - (int(N) - 1) : X(0)) / int(N),
                    (X(1) < 0 ? X(1) - (int(N) - 1) : X(1)) / int(N),
                    (X(2) < 0 ? X(2) - (int(N) - 1) : X(2)) / int(N));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxelf_locate_subvolume(
    const Vector3f &X) {
    return Vector3i(int(floor(X(0) / N)),
                    int(floor(X(1) / N)),
                    int(floor(X(2) / N)));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_global_to_local(
    const Vector3i &X, const Vector3i &Xsv) {
    return Vector3i(X(0) - Xsv(0) * int(N),
                    X(1) - Xsv(1) * int(N),
                    X(2) - Xsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_global_to_local(
    const Vector3f &X, const Vector3i &Xsv) {
    return Vector3f(X(0) - Xsv(0) * int(N),
                    X(1) - Xsv(1) * int(N),
                    X(2) - Xsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_local_to_global(
    const Vector3i &Xlocal, const Vector3i &Xsv) {
    return Vector3i(Xlocal(0) + Xsv(0) * int(N),
                    Xlocal(1) + Xsv(1) * int(N),
                    Xlocal(2) + Xsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxelf_local_to_global(
    const Vector3f &Xlocal, const Vector3i &Xsv) {
    return Vector3f(Xlocal(0) + Xsv(0) * int(N),
                    Xlocal(1) + Xsv(1) * int(N),
                    Xlocal(2) + Xsv(2) * int(N));
}

/** Query **/
template<size_t N>
__device__
UniformTSDFVolumeCudaServer<N> *
ScalableTSDFVolumeCudaServer<N>::QuerySubvolume(
    const Vector3i &Xsv) {
    return hash_table_[Xsv];
}

/** Unoptimized access and interpolation **/
template<size_t N>
__device__
float &ScalableTSDFVolumeCudaServer<N>::tsdf(const Vector3i &X) {
    Vector3i Xsv = voxel_locate_subvolume(X);
    UniformTSDFVolumeCudaServer<N> *subvolume = QuerySubvolume(Xsv);
    return subvolume == nullptr ?
           tsdf_dummy_ : subvolume->tsdf(voxel_global_to_local(X, Xsv));
}

template<size_t N>
__device__
uchar &ScalableTSDFVolumeCudaServer<N>::weight(const Vector3i &X) {
    Vector3i Xsv = voxel_locate_subvolume(X);
    UniformTSDFVolumeCudaServer<N> *subvolume = QuerySubvolume(Xsv);

    return subvolume == nullptr ?
           weight_dummy_ : subvolume->weight(voxel_global_to_local(X, Xsv));
}

template<size_t N>
__device__
Vector3b &ScalableTSDFVolumeCudaServer<N>::color(const Vector3i &X) {
    Vector3i Xsv = voxel_locate_subvolume(X);
    UniformTSDFVolumeCudaServer<N> *subvolume = QuerySubvolume(Xsv);

    return subvolume == nullptr ?
           color_dummy_ : subvolume->color(voxel_global_to_local(X, Xsv));
}

template<size_t N>
__device__
float ScalableTSDFVolumeCudaServer<N>::TSDFAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = Vector3f(X(0) - Xi(0), X(1) - Xi(1), X(2) - Xi(2));

    return (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf(Xi + Vector3i(0, 0, 0)) +
                r(2) * tsdf(Xi + Vector3i(0, 0, 1))
        ) + r(1) * (
            (1 - r(2)) * tsdf(Xi + Vector3i(0, 1, 0)) +
                r(2) * tsdf(Xi + Vector3i(0, 1, 1))
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * tsdf(Xi + Vector3i(1, 0, 0)) +
                r(2) * tsdf(Xi + Vector3i(1, 0, 1))
        ) + r(1) * (
            (1 - r(2)) * tsdf(Xi + Vector3i(1, 1, 0)) +
                r(2) * tsdf(Xi + Vector3i(1, 1, 1))
        ));
}

template<size_t N>
__device__
uchar ScalableTSDFVolumeCudaServer<N>::WeightAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = Vector3f(X(0) - Xi(0), X(1) - Xi(1), X(2) - Xi(2));

    return uchar((1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight(Xi + Vector3i(0, 0, 0)) +
                r(2) * weight(Xi + Vector3i(0, 0, 1))
        ) + r(1) * (
            (1 - r(2)) * weight(Xi + Vector3i(0, 1, 0)) +
                r(2) * weight(Xi + Vector3i(0, 1, 1))
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * weight(Xi + Vector3i(1, 0, 0)) +
                r(2) * weight(Xi + Vector3i(1, 0, 1))
        ) + r(1) * (
            (1 - r(2)) * weight(Xi + Vector3i(1, 1, 0)) +
                r(2) * weight(Xi + Vector3i(1, 1, 1))
        )));
}

template<size_t N>
__device__
Vector3b ScalableTSDFVolumeCudaServer<N>::ColorAt(const Vector3f &X) {
    Vector3i Xi = X.ToVectori();
    Vector3f r = Vector3f(X(0) - Xi(0), X(1) - Xi(1), X(2) - Xi(2));

    Vector3f colorf = (1 - r(0)) * (
        (1 - r(1)) * (
            (1 - r(2)) * color(Xi + Vector3i(0, 0, 0)).ToVectorf() +
                r(2) * color(Xi + Vector3i(0, 0, 1)).ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color(Xi + Vector3i(0, 1, 0)).ToVectorf() +
                r(2) * color(Xi + Vector3i(0, 1, 1)).ToVectorf()
        )) + r(0) * (
        (1 - r(1)) * (
            (1 - r(2)) * color(Xi + Vector3i(1, 0, 0)).ToVectorf() +
                r(2) * color(Xi + Vector3i(1, 0, 1)).ToVectorf()
        ) + r(1) * (
            (1 - r(2)) * color(Xi + Vector3i(1, 1, 0)).ToVectorf() +
                r(2) * color(Xi + Vector3i(1, 1, 1)).ToVectorf()
        ));

    return colorf.ToVectorb();
}

template<size_t N>
__device__
Vector3f ScalableTSDFVolumeCudaServer<N>::GradientAt(
    const Vector3f &X) {
    Vector3f n = Vector3f::Zeros();
    Vector3f X0 = X, X1 = X;

    const float half_gap = voxel_length_;
#pragma unroll 1
    for (size_t k = 0; k < 3; ++k) {
        X0(k) -= half_gap;
        X1(k) += half_gap;

        n(k) = TSDFAt(X1) - TSDFAt(X0);

        X0(k) = X(k);
        X1(k) = X(k);
    }
    return n;
}

/** Optimized access and interpolation **/
template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundary(
    const Vector3i &Xlocal, bool for_gradient) {
    return for_gradient ?
           (Xlocal(0) == 0 || Xlocal(1) == 0 || Xlocal(2) == 0
           || Xlocal(0) >= N - 2 || Xlocal(1) >= N - 2 || Xlocal(2) >= N - 2)
           : (Xlocal(0) == N - 1 || Xlocal(1) == N - 1 || Xlocal(2) == N - 1);
}

template<size_t N>
__device__
inline bool ScalableTSDFVolumeCudaServer<N>::OnBoundaryf(
    const Vector3f &Xlocal, bool for_gradient) {
    return for_gradient ?
           (Xlocal(0) < 1 || Xlocal(1) < 1 || Xlocal(2) < 1
           || Xlocal(0) >= N - 2 || Xlocal(1) >= N - 2 || Xlocal(2) >= N - 2)
           : (Xlocal(0) >= N - 1 || Xlocal(1) >= N - 1 || Xlocal(2) >= N - 1);
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::NeighborOffsetOfBoundaryVoxel(
    const Vector3i &Xlocal) {
    return Vector3i(Xlocal(0) < 0 ? -1 : (Xlocal(0) >= N ? 1 : 0),
                    Xlocal(1) < 0 ? -1 : (Xlocal(1) >= N ? 1 : 0),
                    Xlocal(2) < 0 ? -1 : (Xlocal(2) >= N ? 1 : 0));
}

template<size_t N>
__device__
inline int ScalableTSDFVolumeCudaServer<N>::LinearizeNeighborOffset(
    const Vector3i &dXsv) {
    //return (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
    return 9 * dXsv(2) + 3 * dXsv(1) + dXsv(0) + 13;
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::BoundaryVoxelInNeighbor(
    const Vector3i &Xlocal, const Vector3i &dXsv) {
    return Vector3i(Xlocal(0) - dXsv(0) * int(N),
                    Xlocal(1) - dXsv(1) * int(N),
                    Xlocal(2) - dXsv(2) * int(N));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    const Vector3i &Xlocal,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= Xlocal(0) && Xlocal(0) <= N);
    assert(-1 <= Xlocal(1) && Xlocal(1) <= N);
    assert(-1 <= Xlocal(2) && Xlocal(2) <= N);
#endif
    Vector3f n = Vector3f::Zeros();
    Vector3i X0 = Xlocal, X1 = Xlocal;

#pragma unroll 1
    for (size_t k = 0; k < 3; ++k) {
        X0(k) -= 1;
        X1(k) += 1;

        Vector3i dXsv0 = NeighborOffsetOfBoundaryVoxel(X0);
        Vector3i dXsv1 = NeighborOffsetOfBoundaryVoxel(X1);

        UniformTSDFVolumeCudaServer<N> *subvolume0 =
            cached_subvolumes[LinearizeNeighborOffset(dXsv0)];
        UniformTSDFVolumeCudaServer<N> *subvolume1 =
            cached_subvolumes[LinearizeNeighborOffset(dXsv1)];
        float tsdf0 = (subvolume0 == nullptr) ?
            0 : subvolume0->tsdf(BoundaryVoxelInNeighbor(X0, dXsv0));
        float tsdf1 = (subvolume1 == nullptr) ?
            0 : subvolume1->tsdf(BoundaryVoxelInNeighbor(X1, dXsv1));
        n(k) = tsdf1 - tsdf0;

        X0(k) = X1(k) = Xlocal(k);
    }

    return n;
}

template<size_t N>
__device__
float ScalableTSDFVolumeCudaServer<N>::TSDFOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= Xlocal(0) && Xlocal(0) < N + 1);
    assert(-1 <= Xlocal(1) && Xlocal(1) < N + 1);
    assert(-1 <= Xlocal(2) && Xlocal(2) < N + 1);
#endif

    const Vector3i Xlocali = Xlocal.ToVectori();
    Vector3f r = Vector3f(Xlocal(0) - Xlocali(0),
                          Xlocal(1) - Xlocali(1),
                          Xlocal(2) - Xlocali(2));
    Vector3f rneg = Vector3f(1.0f - r(0), 1.0f - r(1), 1.0f - r(2));

    float sum_weight_interp = 0;
    float sum_tsdf = 0;

    for (size_t k = 0; k < 8; ++k) {
        Vector3i offset_k = Vector3i(shift[k][0], shift[k][1], shift[k][2]);
        Vector3i Xlocali_k = Xlocali + offset_k;

        Vector3i dXsv_k = NeighborOffsetOfBoundaryVoxel(Xlocali_k);
        UniformTSDFVolumeCudaServer<N> *subvolume =
            cached_subvolumes[LinearizeNeighborOffset(dXsv_k)];

        float tsdf_k = (subvolume == nullptr) ? 0.0f :
            subvolume->tsdf(BoundaryVoxelInNeighbor(Xlocali_k, dXsv_k));
        float weight_interp_k = (subvolume == nullptr) ? 0.0f :
            (rneg(0) * (1 - offset_k(0)) + r(0) * offset_k(0)) *
            (rneg(1) * (1 - offset_k(1)) + r(1) * offset_k(1)) *
            (rneg(2) * (1 - offset_k(2)) + r(2) * offset_k(2));

        sum_tsdf += weight_interp_k * tsdf_k;
        sum_weight_interp += weight_interp_k;
    }

    return sum_weight_interp > 0 ? sum_tsdf / sum_weight_interp : 0;
}

template<size_t N>
__device__
uchar ScalableTSDFVolumeCudaServer<N>::WeightOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= Xlocal(0) && Xlocal(0) < N + 1);
    assert(-1 <= Xlocal(1) && Xlocal(1) < N + 1);
    assert(-1 <= Xlocal(2) && Xlocal(2) < N + 1);
#endif

    const Vector3i Xlocali = Xlocal.ToVectori();
    Vector3f r = Vector3f(Xlocal(0) - Xlocali(0),
                          Xlocal(1) - Xlocali(1),
                          Xlocal(2) - Xlocali(2));
    Vector3f rneg = Vector3f(1.0f - r(0), 1.0f - r(1), 1.0f - r(2));

    float sum_weight_interp = 0;
    float sum_weight = 0;
    for (size_t k = 0; k < 8; ++k) {
        Vector3i offset_k = Vector3i(shift[k][0], shift[k][1], shift[k][2]);
        Vector3i Xlocali_k = Xlocali + offset_k;

        Vector3i dXsv_k = NeighborOffsetOfBoundaryVoxel(Xlocali_k);
        UniformTSDFVolumeCudaServer<N> *subvolume =
            cached_subvolumes[LinearizeNeighborOffset(dXsv_k)];

        float weight_k = (subvolume == nullptr) ? 0.0f :
            subvolume->weight(BoundaryVoxelInNeighbor(Xlocali_k, dXsv_k));
        float weight_interp_k = (subvolume == nullptr) ? 0.0f :
            (rneg(0) * (1 - offset_k(0)) + r(0) * offset_k(0)) *
            (rneg(1) * (1 - offset_k(1)) + r(1) * offset_k(1)) *
            (rneg(2) * (1 - offset_k(2)) + r(2) * offset_k(2));

        sum_weight += weight_interp_k * weight_k;
        sum_weight_interp += weight_interp_k;
    }

    return sum_weight_interp > 0 ?
           uchar(fminf(sum_weight / sum_weight_interp, 255)) : uchar(0);
}

template<size_t N>
__device__
Vector3b ScalableTSDFVolumeCudaServer<N>::ColorOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    /** X in range: [-1, N + 1) **/
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(-1 <= Xlocal(0) && Xlocal(0) < N + 1);
    assert(-1 <= Xlocal(1) && Xlocal(1) < N + 1);
    assert(-1 <= Xlocal(2) && Xlocal(2) < N + 1);
#endif

    const Vector3i Xlocali = Xlocal.ToVectori();
    Vector3f r = Vector3f(Xlocal(0) - Xlocali(0),
                          Xlocal(1) - Xlocali(1),
                          Xlocal(2) - Xlocali(2));
    Vector3f rneg = Vector3f(1.0f - r(0), 1.0f - r(1), 1.0f - r(2));

    float sum_weight_interp = 0;
    Vector3f sum_color = Vector3f::Zeros();
    for (size_t k = 0; k < 8; ++k) {
        Vector3i offset_k = Vector3i(shift[k][0], shift[k][1], shift[k][2]);
        Vector3i Xlocali_k = Xlocali + offset_k;

        Vector3i dXsv_k = NeighborOffsetOfBoundaryVoxel(Xlocali_k);
        UniformTSDFVolumeCudaServer<N> *subvolume =
            cached_subvolumes[LinearizeNeighborOffset(dXsv_k)];

        Vector3f color_k = (subvolume == nullptr) ? Vector3f(0) :
            subvolume->color(BoundaryVoxelInNeighbor(Xlocali_k, dXsv_k)).ToVectorf();
        float weight_interp_k = (subvolume == nullptr) ? 0.0f :
            (rneg(0) * (1 - offset_k(0)) + r(0) * offset_k(0)) *
            (rneg(1) * (1 - offset_k(1)) + r(1) * offset_k(1)) *
            (rneg(2) * (1 - offset_k(2)) + r(2) * offset_k(2));

        sum_color += weight_interp_k * color_k;
        sum_weight_interp += weight_interp_k;
    }

    return sum_weight_interp > 0 ?
           (sum_color / sum_weight_interp).ToVectorb() : Vector3b(0);
}

template<size_t N>
__device__
Vector3f ScalableTSDFVolumeCudaServer<N>::GradientOnBoundaryAt(
    const Vector3f &Xlocal,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    Vector3f n = Vector3f::Zeros();
    Vector3f X0 = Xlocal, X1 = Xlocal;

    const float half_gap = voxel_length_;
#pragma unroll 1
    for (size_t k = 0; k < 3; ++k) {
        X0(k) -= half_gap;
        X1(k) += half_gap;
        n(k) = TSDFOnBoundaryAt(X1, cached_subvolumes)
             - TSDFOnBoundaryAt(X0, cached_subvolumes);

        X0(k) = X1(k) = Xlocal(k);
    }
    return n;
}

template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::ActivateSubvolume(
    const HashEntry<Vector3i> &entry) {
    int index = active_subvolume_entry_array_.push_back(entry);
    active_subvolume_indices_[entry.internal_addr] = index;
}

template<size_t N>
__device__
int ScalableTSDFVolumeCudaServer<N>::QueryActiveSubvolumeIndex(
    const Vector3i &key) {
    int internal_addr = hash_table_.GetInternalAddrByKey(key);
    return internal_addr == NULLPTR_CUDA ?
           NULLPTR_CUDA : active_subvolume_indices_[internal_addr];
}

template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::CacheNeighborSubvolumes(
    const Vector3i &Xsv, const Vector3i &dXsv,
    int *cached_subvolume_indices,
    UniformTSDFVolumeCudaServer<N> **cached_subvolumes) {

    Vector3i Xsv_neighbor = Xsv + dXsv;
    int k = LinearizeNeighborOffset(dXsv);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(0 <= k && k < 27);
#endif

    int neighbor_subvolume_idx = QueryActiveSubvolumeIndex(Xsv_neighbor);
    cached_subvolume_indices[k] = neighbor_subvolume_idx;

    /** Some of the subvolumes ARE maintained in hash_table,
     *  but ARE NOT active (NOT in view frustum).
     *  For speed, re-write this part with internal addr accessing.
     *  (can be 0.1 ms faster)
     *  For readablity, keep this. **/
    cached_subvolumes[k] = neighbor_subvolume_idx == NULLPTR_CUDA ?
                           nullptr : QuerySubvolume(Xsv_neighbor);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    if (neighbor_subvolume_idx == NULLPTR_CUDA) {
        assert(cached_subvolumes[k] == nullptr);
    } else {
        HashEntry<Vector3i> &entry =
            active_subvolume_entry_array_[neighbor_subvolume_idx];
        assert(entry.key == Xsv_neighbor);
        assert(hash_table_.GetValuePtrByInternalAddr(entry.internal_addr)
                   == cached_subvolumes[k]);
    }
#endif
}

/** High level functions **/
template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::TouchSubvolume(
    const Vector2i &p,
    ImageCudaServer<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    float d = depth.get(p(0), p(1))(0);

    /** TODO: wrap the criteria in depth image (RGBD-Image) **/
    if (d < 0.1f || d > 3.0f) return;

    Vector3f Xw_near = transform_camera_to_world *
        camera.InverseProjection(p, d - sdf_trunc_);
    Vector3i Xsv_near = voxelf_locate_subvolume(world_to_voxelf(Xw_near));

    Vector3f Xw_far = transform_camera_to_world *
        camera.InverseProjection(p, d + sdf_trunc_);
    Vector3i Xsv_far = voxelf_locate_subvolume(world_to_voxelf(Xw_far));

    /** 3D line from Xsv_near to Xsv_far **/
    /** https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm) **/
    Vector3i DXsv = Xsv_far - Xsv_near;
    Vector3i DXsv_abs = Vector3i(abs(DXsv(0)), abs(DXsv(1)), abs(DXsv(2)));
    int step = DXsv_abs(0) >= DXsv_abs(1) ? DXsv_abs(0) : DXsv_abs(1);
    step = DXsv_abs(2) >= step ? DXsv_abs(2) : step;
    Vector3f DXsv_normalized = DXsv.ToVectorf() * (1.0f / step);

    Vector3f Xsv_curr = Xsv_near.ToVectorf();
    HashEntry<Vector3i> entry;
    for (int k = 0; k <= step; ++k) {
        hash_table_.New(Xsv_curr.ToVectori());
        Xsv_curr += DXsv_normalized;
    }
}

template<size_t N>
__device__
void ScalableTSDFVolumeCudaServer<N>::Integrate(
    const Vector3i &Xlocal,
    HashEntry<Vector3i> &entry,
    ImageCudaServer<open3d::Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    /** Projective data association - additional local to global transform **/
    Vector3f X = voxelf_local_to_global(Xlocal.ToVectorf(), entry.key);
    Vector3f Xw = voxelf_to_world(X);
    Vector3f Xc = transform_camera_to_world.Inverse() * Xw;
    Vector2f p = camera.Projection(Xc);

    /** TSDF **/
    if (!camera.IsValid(p)) return;
    float d = depth.get_interp(p(0), p(1))(0);

    float sdf = Xc(2) - d;
    if (sdf <= -sdf_trunc_) return;
    sdf = fminf(sdf, sdf_trunc_);

    UniformTSDFVolumeCudaServer<N> *subvolume = hash_table_
        .GetValuePtrByInternalAddr(entry.internal_addr);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
    assert(subvolume != nullptr);
#endif

    float &tsdf = subvolume->tsdf(Xlocal);
    uchar &weight = subvolume->weight(Xlocal);
    tsdf = (tsdf * weight + sdf * 1.0f) / (weight + 1.0f);
    weight = uchar(fminf(weight + 1.0f, 255));
}

template<size_t N>
__device__
Vector3f ScalableTSDFVolumeCudaServer<N>::RayCasting(
    const Vector2i &p,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    Vector3f ret = Vector3f(0);

    Vector3f ray_c = camera.InverseProjection(p, 1.0f).normalized();

    /** TODO: throw it into parameters **/
    const float t_min = 0.2f / ray_c(2);
    const float t_max = 3.0f / ray_c(2);

    const Vector3f camera_origin_v = transform_world_to_volume_ *
        (transform_camera_to_world * Vector3f(0));
    const Vector3f ray_v = transform_world_to_volume_.Rotate(
        transform_camera_to_world.Rotate(ray_c));

    float t_prev = 0, tsdf_prev = 0;
    Vector3i Xsv_prev = Vector3i(INT_MIN, INT_MIN, INT_MIN);
    UniformTSDFVolumeCudaServer<N> *subvolume = nullptr;

    /** Do NOT use #pragma unroll: it will make it slow **/
    float t_curr = t_min;
    while (t_curr < t_max) {
        Vector3f Xv_t = camera_origin_v + t_curr * ray_v;
        Vector3i X_t = volume_to_voxelf(Xv_t).ToVectori();
        Vector3i Xsv_t = voxel_locate_subvolume(X_t);
        Vector3i Xlocal_t = voxel_global_to_local(X_t, Xsv_t);

        subvolume = (Xsv_t == Xsv_prev) ? subvolume : QuerySubvolume(Xsv_t);

        float tsdf_curr = subvolume == nullptr ? 0 : subvolume->tsdf(Xlocal_t);
        float step_size = tsdf_curr == 0 ?
                          (subvolume == nullptr ?
                           int(N) * voxel_length_ * 0.8f : sdf_trunc_)
                                         : fmaxf(tsdf_curr, voxel_length_);

        if (tsdf_prev > 0 && tsdf_curr < 0) { /** Zero crossing **/
            float t_intersect = (t_curr * tsdf_prev - t_prev * tsdf_curr)
                / (tsdf_prev - tsdf_curr);

            Vector3f Xv_surface = camera_origin_v + t_intersect * ray_v;
            Vector3f X_surface = volume_to_voxelf(Xv_surface);

            Vector3i Xsv_surface = voxelf_locate_subvolume(X_surface);
            Vector3f Xlocal_surface = voxelf_global_to_local(
                X_surface, Xsv_surface);

            subvolume = (Xsv_t == Xsv_surface) ?
                        subvolume : QuerySubvolume(Xsv_surface);

            Vector3f normal_surface =
                (subvolume == nullptr || OnBoundaryf(Xlocal_surface, true)) ?
                this->GradientAt(X_surface)
                : subvolume->GradientAt(Xlocal_surface);

            return transform_camera_to_world.Inverse().Rotate(
                transform_volume_to_world_.Rotate(normal_surface.normalized()));
        }

        tsdf_prev = tsdf_curr;
        t_prev = t_curr;
        t_curr += step_size;
        Xsv_prev = Xsv_t;
    }

    return ret;
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
    int bucket_count,
    int value_capacity,
    float voxel_length,
    float sdf_trunc,
    TransformCuda &transform_volume_to_world) {

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
    assert(bucket_count > 0 && value_capacity > 0);

    if (server_ != nullptr) {
        PrintError("Already created, stop re-creating!\n");
        return;
    }

    bucket_count_ = bucket_count;
    value_capacity_ = value_capacity;
    server_ = std::make_shared<ScalableTSDFVolumeCudaServer<N>>();
    hash_table_.Create(bucket_count, value_capacity);
    active_subvolume_entry_array_.Create(value_capacity);

    /** Comparing to 512^3, we can hold (sparsely) at most (512^2) 8^3 cubes.
     *  That is 262144. **/
    const int NNN = N * N * N;
    CheckCuda(cudaMalloc(&server_->tsdf_memory_pool_,
                         sizeof(float) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->weight_memory_pool_,
                         sizeof(uchar) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&server_->color_memory_pool_,
                         sizeof(Vector3b) * NNN * value_capacity));

    CheckCuda(cudaMalloc(&server_->active_subvolume_indices_,
                         sizeof(int) * value_capacity));
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
        CheckCuda(cudaFree(server_->active_subvolume_indices_));
    }

    server_ = nullptr;
    hash_table_.Release();
    active_subvolume_entry_array_.Release();
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::UpdateServer() {
    if (server_ != nullptr) {
        server_->hash_table_ = *hash_table_.server();
        server_->active_subvolume_entry_array_ =
            *active_subvolume_entry_array_.server();

        server_->bucket_count_ = bucket_count_;
        server_->value_capacity_ = value_capacity_;

        server_->voxel_length_ = voxel_length_;
        server_->inv_voxel_length_ = 1.0f / voxel_length_;
        server_->sdf_trunc_ = sdf_trunc_;
        server_->transform_volume_to_world_ =
            transform_volume_to_world_;
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
void ScalableTSDFVolumeCuda<N>::TouchSubvolumes(
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
void ScalableTSDFVolumeCuda<N>::IntegrateSubvolumes(
    ImageCuda<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    const int num_blocks = active_subvolume_entry_array_.size();
    const dim3 blocks(num_blocks);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    IntegrateSubvolumesKernel << < blocks, threads >> > (
        *server_, *depth.server(), camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::ResetActiveSubvolumeIndices() {
    CheckCuda(cudaMemset(server_->active_subvolume_indices_, 0xff,
                         sizeof(int) * value_capacity_));
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::Integrate(
    ImageCuda<Vector1f> &depth,
    MonoPinholeCameraCuda &camera,
    TransformCuda &transform_camera_to_world) {

    hash_table_.ResetLocks();
    active_subvolume_entry_array_.set_size(0);
    TouchSubvolumes(depth, camera, transform_camera_to_world);

    ResetActiveSubvolumeIndices();
    GetSubvolumesInFrustum(camera, transform_camera_to_world);
    IntegrateSubvolumes(depth, camera, transform_camera_to_world);
}

template<size_t N>
void ScalableTSDFVolumeCuda<N>::RayCasting(
    ImageCuda<Vector3f> &image,
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