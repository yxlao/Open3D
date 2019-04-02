//
// Created by wei on 10/23/18.
//

#pragma once

#include "ScalableMeshVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {

__global__
void VertexAllocationKernel(
    ScalableMeshVolumeCudaDevice server,
    ScalableTSDFVolumeCudaDevice tsdf_volume) {

    __shared__ UniformTSDFVolumeCudaDevice *neighbor_subvolumes[27];
    __shared__ int neighbor_subvolume_indices[27];

    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        tsdf_volume.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    /** [0, 1, 2] -> [-1, 0, 1], query neighbors **/
    if (Xlocal(0) < 3 && Xlocal(1) < 3 && Xlocal(2) < 3) {
        tsdf_volume.CacheNeighborSubvolumes(
            Xsv, Xlocal - Vector3i::Ones(),
            neighbor_subvolume_indices,
            neighbor_subvolumes);
    }

    __syncthreads();

    if (tsdf_volume.OnBoundary(Xlocal)) {
        server.AllocateVertexOnBoundary(Xlocal, subvolume_idx,
                                        neighbor_subvolume_indices,
                                        neighbor_subvolumes);
    } else {
        server.AllocateVertex(Xlocal, subvolume_idx,
                              neighbor_subvolumes[13]);

    }
}


__host__
void ScalableMeshVolumeCudaKernelCaller::VertexAllocation(
    ScalableMeshVolumeCuda &mesher,
    ScalableTSDFVolumeCuda &tsdf_volume) {

    const dim3 blocks(mesher.active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    VertexAllocationKernel << < blocks, threads >> > (
        *mesher.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}


__global__
void VertexExtractionKernel(
    ScalableMeshVolumeCudaDevice server,
    ScalableTSDFVolumeCudaDevice tsdf_volume) {

    __shared__ UniformTSDFVolumeCudaDevice *neighbor_subvolumes[27];
    __shared__ int neighbor_subvolume_indices[27];

    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        tsdf_volume.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    /** [0, 1, 2] -> [-1, 0, 1], query neighbors **/
    if (Xlocal(0) < 3 && Xlocal(1) < 3 && Xlocal(2) < 3) {
        tsdf_volume.CacheNeighborSubvolumes(
            Xsv, Xlocal - Vector3i::Ones(),
            neighbor_subvolume_indices,
            neighbor_subvolumes);
    }
    __syncthreads();

    if (tsdf_volume.OnBoundary(Xlocal, true)) {
        server.ExtractVertexOnBoundary(Xlocal, subvolume_idx,
                                       Xsv,
                                       tsdf_volume,
                                       neighbor_subvolumes);
    } else {
        server.ExtractVertex(Xlocal, subvolume_idx,
                             Xsv,
                             tsdf_volume,
                             neighbor_subvolumes[13]);
    }
}


__host__
void ScalableMeshVolumeCudaKernelCaller::VertexExtraction(
    ScalableMeshVolumeCuda &mesher,
    ScalableTSDFVolumeCuda &tsdf_volume) {

    const dim3 blocks(mesher.active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    VertexExtractionKernel << < blocks, threads >> > (
        *mesher.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

}


__global__
void TriangleExtractionKernel(
    ScalableMeshVolumeCudaDevice server,
    ScalableTSDFVolumeCudaDevice tsdf_volume) {

    __shared__ UniformTSDFVolumeCudaDevice *neighbor_subvolumes[27];
    __shared__ int neighbor_subvolume_indices[27];

    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        tsdf_volume.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    /** [0, 1, 2] -> [-1, 0, 1], query neighbors **/
    if (Xlocal(0) < 3 && Xlocal(1) < 3 && Xlocal(2) < 3) {
        tsdf_volume.CacheNeighborSubvolumes(
            Xsv, Xlocal - Vector3i::Ones(),
            neighbor_subvolume_indices,
            neighbor_subvolumes);
    }
    __syncthreads();

    if (tsdf_volume.OnBoundary(Xlocal)) {
        server.ExtractTriangleOnBoundary(Xlocal, subvolume_idx,
                                         neighbor_subvolume_indices);
    } else {
        server.ExtractTriangle(Xlocal, subvolume_idx);
    }
}


void ScalableMeshVolumeCudaKernelCaller::TriangleExtraction(
    ScalableMeshVolumeCuda &mesher,
    ScalableTSDFVolumeCuda &tsdf_volume) {

    const dim3 blocks(mesher.active_subvolumes_);
    const dim3 threads(THREAD_3D_UNIT, THREAD_3D_UNIT, THREAD_3D_UNIT);
    TriangleExtractionKernel << < blocks, threads >> > (
        *mesher.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d