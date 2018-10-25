//
// Created by wei on 10/23/18.
//

#pragma once

#include "ScalableMeshVolumeCuda.cuh"

namespace open3d {
template<VertexType type, size_t N>
__global__
void MarchingCubesVertexAllocationKernel(
    ScalableMeshVolumeCudaServer<type, N> server,
    ScalableTSDFVolumeCudaServer<N> tsdf_volume) {

    __shared__ UniformTSDFVolumeCudaServer<N> *neighbor_active_subvolumes[27];
    __shared__ int neighbor_active_subvolume_indices[27];

    const int subvolume_idx = blockIdx.x;
    const HashEntry<Vector3i> &subvolume_entry =
        tsdf_volume.active_subvolume_entry_array()[subvolume_idx];

    const Vector3i Xsv = subvolume_entry.key;
    const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y, threadIdx.z);

    if (Xlocal(0) < 3 && Xlocal(1) < 3 && Xlocal(2) < 3) {
        /** [0, 1, 2] -> [-1, 0, 1], query neighbors **/
        Vector3i dXsv = Xlocal - Vector3i::Ones();

        Vector3i Xsv_neighbor = Xsv + dXsv;
        int neighbor_idx = tsdf_volume.LinearizeNeighborIndex(dXsv);

        int active_subvolume_index = tsdf_volume
            .QueryActiveSubvolumeIndex(Xsv_neighbor);

        neighbor_active_subvolume_indices[neighbor_idx] =
            active_subvolume_index;

        /** Some of the subvolumes ARE maintained in hash_table,
         *  but ARE NOT active (NOT in view frustum).
         *  For speed, re-write this part with internal addr accessing.
         *  (can be 0.1 ms faster)
         *  For readablity, keep this. **/
        neighbor_active_subvolumes[neighbor_idx] =
            active_subvolume_index == NULLPTR_CUDA ?
            nullptr : tsdf_volume.QuerySubvolume(Xsv_neighbor);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        if (active_subvolume_index == NULLPTR_CUDA) {
            assert(neighbor_active_subvolumes[neighbor_idx] == nullptr);
        } else {
            HashEntry<Vector3i> &entry = tsdf_volume
                .active_subvolume_entry_array()[active_subvolume_index];
            assert(Xsv_neighbor == entry.key);
            assert(neighbor_active_subvolumes[neighbor_idx]
            == tsdf_volume.hash_table().GetValuePtrByInternalAddr(
                entry.internal_addr));
        }
#endif
    }
    __syncthreads();

    server.AllocateVertex(Xlocal(0), Xlocal(1), Xlocal(2), subvolume_idx,
                          tsdf_volume, neighbor_active_subvolumes);
}

template<VertexType type, size_t N>
__global__
void MarchingCubesVertexExtractionKernel(
    ScalableMeshVolumeCudaServer<type, N> server,
    ScalableTSDFVolumeCudaServer<N> tsdf_volume) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    __shared__ UniformTSDFVolumeCudaServer<N> *neighbor_subvolumes[27];

    const int subvolume_idx = blockIdx.x;
    if (subvolume_idx >= tsdf_volume.active_subvolume_entry_array().size())
        return;

    const int xlocal = threadIdx.x;
    const int ylocal = threadIdx.y;
    const int zlocal = threadIdx.z;

    //server.ExtractVertex(x, y, z, tsdf_volume);
}

template<VertexType type, size_t N>
__global__
void MarchingCubesTriangleExtractionKernel(
    ScalableMeshVolumeCudaServer<type, N> server) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= N - 1 || y >= N - 1 || z >= N - 1) return;

    //server.ExtractTriangle(x, y, z);
}
}