//
// Created by wei on 10/12/18.
//

#pragma once

#include "PointCloudCuda.h"
#include <Cuda/Container/ArrayCudaDevice.cuh>

namespace open3d {
namespace cuda {
__global__
void BuildFromRGBDImageKernel(PointCloudCudaDevice pcl,
                              RGBDImageCudaDevice rgbd,
                              PinholeCameraIntrinsicCuda intrinsic) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= rgbd.width_ || y >= rgbd.height_) return;

    float depth = rgbd.depth_.at(x, y)(0);
    if (depth == 0) return;
    Vector3b color = rgbd.color_.at(x, y);

    Vector3f point = intrinsic.InverseProjectPixel(Vector2i(x, y), depth);

    int index = pcl.points_.push_back(point);
    if (pcl.type_ & VertexWithColor) {
        pcl.colors_[index] = color.ToVectorf() / 255.0f;
    }
}

__host__
void PointCloudCudaKernelCaller::BuildFromRGBDImage(
    PointCloudCuda &pcl, RGBDImageCuda &rgbd,
    PinholeCameraIntrinsicCuda &intrinsic) {
    const dim3 blocks(DIV_CEILING(rgbd.width_, THREAD_2D_UNIT),
                      DIV_CEILING(rgbd.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BuildFromRGBDImageKernel << < blocks, threads >> > (
        *pcl.device_, *rgbd.device_, intrinsic);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void BuildFromDepthImageKernel(PointCloudCudaDevice pcl,
                               ImageCudaDevice<Vector1f> depth,
                               PinholeCameraIntrinsicCuda intrinsic) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.width_ || y >= depth.height_) return;

    float d = depth.at(x, y)(0);
    if (d == 0) return;

    Vector3f point = intrinsic.InverseProjectPixel(Vector2i(x, y), d);
    pcl.points_.push_back(point);
}

__host__
void PointCloudCudaKernelCaller::BuildFromDepthImage(
    PointCloudCuda &pcl, ImageCuda<Vector1f> &depth,
    PinholeCameraIntrinsicCuda &intrinsic) {
    const dim3 blocks(DIV_CEILING(depth.width_, THREAD_2D_UNIT),
                      DIV_CEILING(depth.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    BuildFromDepthImageKernel << < blocks, threads >> > (
        *pcl.device_, *depth.device_, intrinsic);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

/** Duplicate of TriangleMesh ... anyway to simplify it? **/
__global__
void GetMinBoundKernel(PointCloudCudaDevice pcl,
                       ArrayCudaDevice<Vector3f> min_bound) {
    __shared__ float local_min_x[THREAD_1D_UNIT];
    __shared__ float local_min_y[THREAD_1D_UNIT];
    __shared__ float local_min_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < pcl.points_.size() ?
                      pcl.points_[idx] : Vector3f(1e10f);

    local_min_x[tid] = vertex(0);
    local_min_y[tid] = vertex(1);
    local_min_z[tid] = vertex(2);
    __syncthreads();

    if (tid < 128) {
        local_min_x[tid] = fminf(local_min_x[tid], local_min_x[tid + 128]);
        local_min_y[tid] = fminf(local_min_y[tid], local_min_y[tid + 128]);
        local_min_z[tid] = fminf(local_min_z[tid], local_min_z[tid + 128]);
    }
    __syncthreads();

    if (tid < 64) {
        local_min_x[tid] = fminf(local_min_x[tid], local_min_x[tid + 64]);
        local_min_y[tid] = fminf(local_min_y[tid], local_min_y[tid + 64]);
        local_min_z[tid] = fminf(local_min_z[tid], local_min_z[tid + 64]);
    }
    __syncthreads();

    if (tid < 32) {
        WarpReduceMin(local_min_x, tid);
        WarpReduceMin(local_min_y, tid);
        WarpReduceMin(local_min_z, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicMinf(&(min_bound[0](0)), local_min_x[0]);
        atomicMinf(&(min_bound[0](1)), local_min_y[0]);
        atomicMinf(&(min_bound[0](2)), local_min_z[0]);
    }
}

__host__
void PointCloudCudaKernelCaller::GetMinBound(
    const PointCloudCuda &pcl, ArrayCuda<Vector3f> &min_bound) {
    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMinBoundKernel << < blocks, threads >> >(
        *pcl.device_, *min_bound.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void GetMaxBoundKernel(PointCloudCudaDevice pcl,
                       ArrayCudaDevice<Vector3f> max_bound) {
    __shared__ float local_max_x[THREAD_1D_UNIT];
    __shared__ float local_max_y[THREAD_1D_UNIT];
    __shared__ float local_max_z[THREAD_1D_UNIT];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    Vector3f vertex = idx < pcl.points_.size() ?
                      pcl.points_[idx] : Vector3f(-1e10f);

    local_max_x[tid] = vertex(0);
    local_max_y[tid] = vertex(1);
    local_max_z[tid] = vertex(2);
    __syncthreads();

    if (tid < 128) {
        local_max_x[tid] = fmaxf(local_max_x[tid], local_max_x[tid + 128]);
        local_max_y[tid] = fmaxf(local_max_y[tid], local_max_y[tid + 128]);
        local_max_z[tid] = fmaxf(local_max_z[tid], local_max_z[tid + 128]);
        __syncthreads();
    }

    if (tid < 64) {
        local_max_x[tid] = fmaxf(local_max_x[tid], local_max_x[tid + 64]);
        local_max_y[tid] = fmaxf(local_max_y[tid], local_max_y[tid + 64]);
        local_max_z[tid] = fmaxf(local_max_z[tid], local_max_z[tid + 64]);
        __syncthreads();
    }

    if (tid < 32) {
        WarpReduceMax(local_max_x, tid);
        WarpReduceMax(local_max_y, tid);
        WarpReduceMax(local_max_z, tid);
    }
    __syncthreads();

    if (tid == 0) {
        atomicMaxf(&(max_bound[0](0)), local_max_x[0]);
        atomicMaxf(&(max_bound[0](1)), local_max_y[0]);
        atomicMaxf(&(max_bound[0](2)), local_max_z[0]);
    }
}

__host__
void PointCloudCudaKernelCaller::GetMaxBound(
    const PointCloudCuda &pcl, ArrayCuda<Vector3f> &max_bound) {

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetMaxBoundKernel << < blocks, threads >> > (
        *pcl.device_, *max_bound.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void TransformKernel(PointCloudCudaDevice pcl, TransformCuda transform) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= pcl.points_.size()) return;

    Vector3f &position = pcl.points_[idx];
    position = transform * position;

    if (pcl.type_ & VertexWithNormal) {
        Vector3f &normal = pcl.normals_[idx];
        normal = transform.Rotate(normal);
    }
}

__host__
void PointCloudCudaKernelCaller::Transform(
    PointCloudCuda &pcl, TransformCuda &transform) {

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    TransformKernel << < blocks, threads >> > (*pcl.device_, transform);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d