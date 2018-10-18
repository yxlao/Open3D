//
// Created by wei on 10/10/18.
//

#pragma once

#include "ScalableTSDFVolumeCuda.h"

namespace open3d {
/** Coordinate conversions **/
template<size_t N>
__device__
inline Vector3i
ScalableTSDFVolumeCudaServer<N>::Vectorize(size_t index) {
    Vector3i ret;
    ret(0) = int(index % N);
    ret(1) = int((index % (N * N)) / N);
    ret(2) = int(index / (N * N));
    return ret;
}

template<size_t N>
__device__
inline int ScalableTSDFVolumeCudaServer<N>::IndexOf(int x, int y, int z) {
    return int(z * (N * N) + y * N + x);
}

template<size_t N>
__device__
inline int ScalableTSDFVolumeCudaServer<N>::IndexOf(const Vector3i &X) {
    return IndexOf(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::world_to_voxel(
    float x, float y, float z) {
    return world_to_voxel(Vector3f(x, y, z));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::world_to_voxel(
    const Vector3f &X) {
    /** Coordinate transform **/
    return volume_to_voxel(transform_world_to_volume_ * X);
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
    const Vector3f &X_v) {
    return transform_volume_to_world_ * voxel_to_volume(X_v);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_to_volume(
    float x, float y, float z) {
    return Vector3f((x + 0.5f) * voxel_length_, /** Scale transform **/
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
    float x, float y, float z) {
    return Vector3f(x * inv_voxel_length_ - 0.5f,
                    y * inv_voxel_length_ - 0.5f,
                    z * inv_voxel_length_ - 0.5f);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::volume_to_voxel(
    const Vector3f &X) {
    return volume_to_voxel(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_to_block(
    float x, float y, float z) {
    return Vector3i(int(floor(x / N)), int(floor(y / N)), int(floor(z / N)));
}

template<size_t N>
__device__
inline Vector3i ScalableTSDFVolumeCudaServer<N>::voxel_to_block(
    const Vector3f &X) {
    return voxel_to_block(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_in_block(
    const Vector3f &X, const Vector3i &block) {
    return Vector3f(X(0) - block(0) * N,
                    X(1) - block(1) * N,
                    X(2) - block(2) * N);
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::voxel_in_block(
    float x, float y, float z, int block_x, int block_y, int block_z) {
    return Vector3f(x - block_x * N,
                    y - block_y * N,
                    z - block_z * N);
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::tsdf(int x, int y, int z) {
    Vector3i block_X = voxel_to_block(x, y, z);
    Vector3f voxel_in_block_X = voxel_in_block(x, y, z,
        block_X(0), block_X(1), block_X(2));

    UniformTSDFVolumeCudaServer *block = hash_table_[block_X];

    return block == nullptr ?
           0 : block->tsdf(int(voxel_in_block_X(0)),
                           int(voxel_in_block_X(1)),
                           int(voxel_in_block_X(2)));
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::tsdf(const Vector3i& X) {
    return tsdf(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline uchar ScalableTSDFVolumeCudaServer<N>::weight(int x, int y, int z) {
    Vector3i block_X = voxel_to_block(x, y, z);
    Vector3f voxel_in_block_X = voxel_in_block(x, y, z,
        block_X(0), block_X(1), block_X(2));

    UniformTSDFVolumeCudaServer *block = hash_table_[block_X];

    return block == nullptr ?
           0 : block->weight(int(voxel_in_block_X(0)),
                             int(voxel_in_block_X(1)),
                             int(voxel_in_block_X(2)));
}

template<size_t N>
__device__
inline uchar ScalableTSDFVolumeCudaServer<N>::weight(const Vector3i &X) {
    return weight(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3b ScalableTSDFVolumeCudaServer<N>::color(int x, int y, int z) {
    Vector3i block_X = voxel_to_block(x, y, z);
    Vector3f voxel_in_block_X = voxel_in_block(x, y, z,
        block_X(0), block_X(1), block_X(2));

    UniformTSDFVolumeCudaServer *block = hash_table_[block_X];

    return block == nullptr ?
           Vector3b(0) : block->color(int(voxel_in_block_X(0)),
                                      int(voxel_in_block_X(1)),
                                      int(voxel_in_block_X(2)));
}

template<size_t N>
__device__
inline Vector3b ScalableTSDFVolumeCudaServer<N>::color(const Vector3i &X) {
    return color(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    int x, int y, int z) {
    return Vector3f(tsdf(x + 1, y, z) - tsdf(x - 1, y, z),
                    tsdf(x, y + 1, z) - tsdf(x, y - 1, z),
                    tsdf(x, y, z + 1) - tsdf(x, y, z - 1));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    const Vector3i& X) {
    return graident(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::gradient(
    const Vector3f &X) {
    return gradient(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::TSDFAt(
    float x, float y, float z) {
    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    float weight_sum = 0;
    float tsdf_sum = 0;
    for (int i = 0; i < 8; ++i) {
        Vector3i offset_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + offset_i;

        uchar weight_tsdf = weight(X_i);

        float weight_interp = weight_tsdf == 0 ?
            0.0f
            : ((1 - offset_x) * (1 - r(0)) + offset_x * r(0))
            * ((1 - offset_y) * (1 - r(1)) + offset_y * r(1))
            * ((1 - offset_z) * (1 - r(2)) + offset_z * r(2));
        weight_sum += weight_interp;
        tsdf_sum += weight_tsdf == 0 ? 0 : weight_interp * tsdf(X_i);
    }

    return weight_sum == 0 ? 0 : tsdf_sum / weight_sum;
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::TSDFAt(const Vector3f& X) {
    return TSDFAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::WeightAt(
    float x, float y, float z) {
    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    float weight_sum = 0;
    float weight_tsdf_sum = 0;
    for (int i = 0; i < 8; ++i) {
        Vector3i offset_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + offset_i;

        uchar weight_tsdf = weight(X_i);
        float weight_interp = weight_tsdf == 0 ?
                      0.0f
                      : ((1 - offset_x) * (1 - r(0)) + offset_x * r(0))
                      * ((1 - offset_y) * (1 - r(1)) + offset_y * r(1))
                      * ((1 - offset_z) * (1 - r(2)) + offset_z * r(2));

        weight_sum += weight_interp;
        weight_tsdf_sum += weight_interp * weight_tsdf;
    }

    return weight_sum == 0 ? 0 : uchar(weight_tsdf_sum / weight_sum);
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::WeightAt(const Vector3f &X) {
    return WeightAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::ColorAt(
    float x, float y, float z) {
    Vector3f Xf(x, y, z);
    Vector3i X = Vector3i(int(x), int(y), int(z));
    Vector3f r = Xf - X.ToVectorf();

    float weight_sum = 0;
    Vector3f color_sum = 0;
    for (int i = 0; i < 8; ++i) {
        Vector3i offset_i = Vector3i((i & 4) >> 2, (i & 2) >> 1, i & 1);
        Vector3i X_i = X + offset_i;

        uchar weight_tsdf = weight(X_i);
        float weight_interp = weight_tsdf == 0 ?
                      0.0f
                      : ((1 - offset_x) * (1 - r(0)) + offset_x * r(0))
                      * ((1 - offset_y) * (1 - r(1)) + offset_y * r(1))
                      * ((1 - offset_z) * (1 - r(2)) + offset_z * r(2));
        color_sum += weight_tsdf == 0 ? 0.0f : weight_interp * color(X_i);
    }

    return weight_sum == 0 ? 0 : Vector3b(uchar(color_sum(0) / weight_sum),
                                          uchar(color_sum(1) / weight_sum),
                                          uchar(color_sum(2) / weight_sum));
}

template<size_t N>
__device__
inline float ScalableTSDFVolumeCudaServer<N>::ColorAt(const Vector3f& X) {
    return ColorAt(X(0), X(1), X(2));
}

template<size_t N>
__device__
inline Vector3f ScalableTSDFVolumeCudaServer<N>::GradientAt(const Vector3f &X) {
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

}