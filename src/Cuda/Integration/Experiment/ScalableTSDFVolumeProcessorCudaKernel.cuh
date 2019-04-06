//
// Created by wei on 10/23/18.
//

#pragma once

#include "ScalableTSDFVolumeProcessorCuda.h"
#include <Cuda/Integration/ScalableTSDFVolumeCudaDevice.cuh>
#include <Cuda/Integration/UniformTSDFVolumeCudaDevice.cuh>

namespace open3d {
namespace cuda {

__global__
void ComputeGradientKernel(
    ScalableTSDFVolumeProcessorCudaDevice server,
    ScalableTSDFVolumeCudaDevice tsdf_volume) {

    __shared__ UniformTSDFVolumeCudaDevice *neighbor_subvolumes[27];
    __shared__ int neighbor_subvolume_indices[27];

    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        tsdf_volume.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    /** query 27 neighbors **/
    if (Xlocal(0) < 3 && Xlocal(1) < 3 && Xlocal(2) < 3) {
        tsdf_volume.CacheNeighborSubvolumes(
            Xsv, Xlocal - Vector3i::Ones(),
            neighbor_subvolume_indices,
            neighbor_subvolumes);
    }
    __syncthreads();

    if (neighbor_subvolumes[13]->weight(Xlocal) == 0) {
        server.gradient(Xlocal, subvolume_idx) = Vector3f(0);
    } else {
        server.gradient(Xlocal, subvolume_idx)
            = tsdf_volume.OnBoundary(Xlocal, true)
              ? neighbor_subvolumes[13]->gradient(Xlocal)
              : tsdf_volume.gradient(Xlocal, neighbor_subvolumes);
    }
}

__host__
void ScalableGradientVolumeCudaKernelCaller::ComputeGradient(
    ScalableTSDFVolumeProcessorCuda &server,
    ScalableTSDFVolumeCuda &tsdf_volume) {

    const dim3 blocks(server.active_subvolumes_);
    const dim3 threads(server.N_, server.N_, server.N_);
    ComputeGradientKernel << < blocks, threads >> > (
        *server.device_, *tsdf_volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ExtractVoxelsNearSurfaceKernel(ScalableTSDFVolumeProcessorCudaDevice server,
                                    ScalableTSDFVolumeCudaDevice volume,
                                    PointCloudCudaDevice pcl,
                                    float threshold) {
    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        volume.active_subvolume_entry_array_[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    __shared__ UniformTSDFVolumeCudaDevice *subvolume;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        subvolume = volume.QuerySubvolume(subvolume_entry.key);
    }
    __syncthreads();

    float tsdf = subvolume->tsdf(Xlocal);
    uchar weight = subvolume->weight(Xlocal);
    if (weight > 0 && fabsf(tsdf) <= threshold) {
        Vector3i Xglobal = volume.voxel_local_to_global(Xlocal, Xsv);
        Vector3f Xworld = volume.voxelf_to_world(Xglobal.cast<float>());

        int addr = pcl.points_.push_back(Xworld);
        pcl.colors_[addr] = Jet(tsdf, -0.5f, 0.5f);
        pcl.normals_[addr] = server.gradient(Xlocal, subvolume_idx);
    }
}

void ScalableGradientVolumeCudaKernelCaller::ExtractVoxelsNearSurfaces(
    ScalableTSDFVolumeProcessorCuda &server,
    ScalableTSDFVolumeCuda &volume,
    PointCloudCuda &pcl,
    float threshold){

    const dim3 blocks(volume.active_subvolume_entry_array_.size());
    const dim3 threads(volume.N_, volume.N_, volume.N_);
    ExtractVoxelsNearSurfaceKernel<< < blocks, threads >> > (
        *server.device_, *volume.device_, *pcl.device_, threshold);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
} // cuda
} // open3d