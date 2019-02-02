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

    TripleBlockReduceMin<float>(local_min_x, local_min_y, local_min_z, tid);

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

    TripleBlockReduceMax<float>(local_max_x, local_max_y, local_max_z, tid);

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
void ComputeSumKernel(PointCloudCudaDevice device,
                      ArrayCudaDevice<Vector3f> sum) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= device.points_.size()) return;

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    Vector3f &vertex = device.points_[idx];
    local_sum0[tid] = vertex(0);
    local_sum1[tid] = vertex(1);
    local_sum2[tid] = vertex(2);
    __syncthreads();

    TripleBlockReduceSum<float>(local_sum0, local_sum1, local_sum2, tid);

    if (tid == 0) {
        atomicAdd(&sum[0](0), local_sum0[0]);
        atomicAdd(&sum[0](1), local_sum1[0]);
        atomicAdd(&sum[0](2), local_sum2[0]);
    }
    __syncthreads();
}

void PointCloudCudaKernelCaller::ComputeSum(
    PointCloudCuda &pcl, ArrayCuda<Vector3f> &sum) {

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ComputeSumKernel<<<blocks, threads>>>(*pcl.device_, *sum.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void SubMeanAndGetMaxScaleKernel(PointCloudCudaDevice device,
                                Vector3f mean,
                                ArrayCudaDevice<float> scale) {
    __shared__ float local_max[THREAD_1D_UNIT];

    const int tid = threadIdx.x;
    local_max[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= device.points_.size()) return;

    Vector3f &vertex = device.points_[idx];
    vertex -= mean;

    local_max[tid] = vertex.norm();
    __syncthreads();

    BlockReduceMax<float>(local_max, tid);

    if (tid == 0) {
        atomicMaxf(&scale[0], local_max[0]);
    }
}

void PointCloudCudaKernelCaller::Normalize(
    PointCloudCuda &pcl, const Vector3f &mean,
    ArrayCuda<float> &max_scale) {

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    SubMeanAndGetMaxScaleKernel <<<blocks, threads>>>(
        *pcl.device_, mean, *max_scale.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void RescaleKernel(PointCloudCudaDevice device, float scale) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= device.points_.size()) return;

    Vector3f &vertex = device.points_[idx];
    vertex /= scale;
}

void PointCloudCudaKernelCaller::Rescale(PointCloudCuda &pcl, float scale){
    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    RescaleKernel<<<blocks, threads>>>(*pcl.device_, scale);
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