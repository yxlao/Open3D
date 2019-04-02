//
// Created by wei on 11/9/18.
//

#pragma once

#include "ScalableMeshVolumeCuda.h"
#include "MarchingCubesConstCuda.h"

#include "ScalableTSDFVolumeCuda.h"

#include <cuda_runtime.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 */

ScalableMeshVolumeCuda::ScalableMeshVolumeCuda() {
    N_ = -1;
    max_subvolumes_ = -1;
    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}


ScalableMeshVolumeCuda::ScalableMeshVolumeCuda(
    VertexType type,
    int N, int max_subvolumes,
    int max_vertices, int max_triangles) {

    Create(type, N, max_subvolumes, max_vertices, max_triangles);
}


ScalableMeshVolumeCuda::ScalableMeshVolumeCuda(
    const ScalableMeshVolumeCuda &other) {
    N_ = other.N_;
    max_subvolumes_ = other.max_subvolumes_;

    vertex_type_ = other.vertex_type_;
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    device_ = other.device_;
    mesh_ = other.mesh();
}


ScalableMeshVolumeCuda &ScalableMeshVolumeCuda::operator=(
    const ScalableMeshVolumeCuda &other) {
    if (this != &other) {
        N_ = other.N_;
        max_subvolumes_ = other.max_subvolumes_;

        vertex_type_ = other.vertex_type_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        device_ = other.device_;
        mesh_ = other.mesh();
    }
    return *this;
}


ScalableMeshVolumeCuda::~ScalableMeshVolumeCuda() {
    Release();
}


void ScalableMeshVolumeCuda::Create(
    VertexType type, int N, int max_subvolumes,
    int max_vertices, int max_triangles) {
    if (device_ != nullptr) {
        utility::PrintError("[ScalableMeshVolumeCuda]: "
                            "Already created, abort!\n");
        return;
    }

    assert(N_ > 0 && max_subvolumes > 0 && max_vertices > 0 && max_triangles > 0);

    device_ = std::make_shared<ScalableMeshVolumeCudaDevice>();

    N_ = N;
    max_subvolumes_ = max_subvolumes;

    vertex_type_ = type;
    max_vertices_ = max_vertices;
    max_triangles_ = max_triangles;

    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMalloc(&device_->table_indices_memory_pool_,
                         sizeof(uchar) * NNN * max_subvolumes_));
    CheckCuda(cudaMalloc(&device_->vertex_indices_memory_pool_,
                         sizeof(Vector3i) * NNN * max_subvolumes_));
    mesh_.Create(type, max_vertices_, max_triangles_);

    UpdateDevice();
    Reset();
}


void ScalableMeshVolumeCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->table_indices_memory_pool_));
        CheckCuda(cudaFree(device_->vertex_indices_memory_pool_));
    }

    mesh_.Release();
    device_ = nullptr;
    max_subvolumes_ = -1;
    max_vertices_ = -1;
    max_triangles_ = -1;
}


void ScalableMeshVolumeCuda::Reset() {
    if (device_ != nullptr) {
        const size_t NNN = N_ * N_ * N_;
        CheckCuda(cudaMemset(device_->table_indices_memory_pool_, 0,
                             sizeof(uchar) * NNN * max_subvolumes_));
        CheckCuda(cudaMemset(device_->vertex_indices_memory_pool_, 0,
                             sizeof(Vector3i) * NNN * max_subvolumes_));
        mesh_.Reset();
    }
}


void ScalableMeshVolumeCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->N_ = N_;
        device_->mesh_ = *mesh_.device_;
    }
}


void ScalableMeshVolumeCuda::VertexAllocation(
    ScalableTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller::VertexAllocation(*this, tsdf_volume);

    timer.Stop();
    utility::PrintDebug("Allocation takes %f milliseconds\n", timer.GetDuration());
}


void ScalableMeshVolumeCuda::VertexExtraction(
    ScalableTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller::VertexExtraction(*this, tsdf_volume);

    timer.Stop();
    utility::PrintDebug("Extraction takes %f milliseconds\n", timer.GetDuration
    ());
}


void ScalableMeshVolumeCuda::TriangleExtraction(
    ScalableTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    ScalableMeshVolumeCudaKernelCaller::TriangleExtraction(
        *this, tsdf_volume);

    timer.Stop();
    utility::PrintDebug("Triangulation takes %f milliseconds\n", timer
    .GetDuration());
}


void ScalableMeshVolumeCuda::MarchingCubes(
    ScalableTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr && vertex_type_ != VertexTypeUnknown);

    mesh_.Reset();
    active_subvolumes_ = tsdf_volume.active_subvolume_entry_array_.size();
    utility::PrintDebug("Active subvolumes: %d\n", active_subvolumes_);

    if (active_subvolumes_ <= 0) {
        utility::PrintError("Invalid active subvolume numbers: %d !\n",
                   active_subvolumes_);
        return;
    }

    VertexAllocation(tsdf_volume);
    VertexExtraction(tsdf_volume);

    TriangleExtraction(tsdf_volume);

    if (vertex_type_ & VertexWithNormal) {
        mesh_.vertex_normals_.set_iterator(mesh_.vertices_.size());
    }
    if (vertex_type_ & VertexWithColor) {
        mesh_.vertex_colors_.set_iterator(mesh_.vertices_.size());
    }
}
} // cuda
} // open3d