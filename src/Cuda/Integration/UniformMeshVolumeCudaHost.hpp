//
// Created by wei on 11/9/18.
//


#include "UniformMeshVolumeCuda.h"
#include "MarchingCubesConstCuda.h"

#include <cuda_runtime.h>

namespace open3d {
namespace cuda {
/**
 * Client end
 */

UniformMeshVolumeCuda::UniformMeshVolumeCuda() {
    N_ = -1;
    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}


UniformMeshVolumeCuda::UniformMeshVolumeCuda(
    VertexType type, int N, int max_vertices, int max_triangles) {
    Create(type, N, max_vertices, max_triangles);
}


UniformMeshVolumeCuda::UniformMeshVolumeCuda(
    const UniformMeshVolumeCuda &other) {
    N_ = other.N_;
    vertex_type_ = other.vertex_type_;
    max_vertices_ = other.max_vertices_;
    max_triangles_ = other.max_triangles_;

    device_ = other.device_;
    mesh_ = other.mesh();
}


UniformMeshVolumeCuda &UniformMeshVolumeCuda::operator=(
    const UniformMeshVolumeCuda &other) {
    if (this != &other) {
        N_ = other.N_;
        vertex_type_ = other.vertex_type_;
        max_vertices_ = other.max_vertices_;
        max_triangles_ = other.max_triangles_;

        device_ = other.device_;
        mesh_ = other.mesh();
    }
    return *this;
}


UniformMeshVolumeCuda::~UniformMeshVolumeCuda() {
    Release();
}


void UniformMeshVolumeCuda::Create(
    VertexType type, int N, int max_vertices, int max_triangles) {
    if (device_ != nullptr) {
        utility::PrintError("[UniformMeshVolumeCuda] Already created, "
                            "abort!\n");
        return;
    }

    assert(max_vertices > 0 && max_triangles > 0);
    assert(type != VertexTypeUnknown);

    device_ = std::make_shared<UniformMeshVolumeCudaDevice>();

    N_ = N;
    vertex_type_ = type;
    max_triangles_ = max_triangles;
    max_vertices_ = max_vertices;

    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMalloc(&device_->table_indices_, sizeof(uchar) * NNN));
    CheckCuda(cudaMalloc(&device_->vertex_indices_, sizeof(Vector3i) * NNN));
    mesh_.Create(vertex_type_, max_vertices_, max_triangles_);

    UpdateDevice();
    Reset();
}


void UniformMeshVolumeCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->table_indices_));
        CheckCuda(cudaFree(device_->vertex_indices_));
    }
    mesh_.Release();
    device_ = nullptr;

    vertex_type_ = VertexTypeUnknown;
    max_vertices_ = -1;
    max_triangles_ = -1;
}

/** Reset only have to be performed once on initialization:
 * 1. table_indices_ will be reset in kernels;
 * 2. None of the vertex_indices_ will be -1 after this reset, because
 *  - The not effected vertex indices will remain 0;
 *  - The effected vertex indices will be >= 0 after being assigned address.
 **/

void UniformMeshVolumeCuda::Reset() {
    if (device_ != nullptr) {
        const size_t NNN = N_ * N_ * N_;
        CheckCuda(cudaMemset(device_->table_indices_, 0,
                             sizeof(uchar) * NNN));
        CheckCuda(cudaMemset(device_->vertex_indices_, 0,
                             sizeof(Vector3i) * NNN));
        mesh_.Reset();
    }
}


void UniformMeshVolumeCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->N_ = N_;
        device_->mesh_ = *mesh_.device_;
    }
}


void UniformMeshVolumeCuda::VertexAllocation(
    UniformTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller::VertexAllocation(*this, tsdf_volume);

    timer.Stop();
    utility::PrintInfo("Allocation takes %f milliseconds\n", timer.GetDuration
    ());
}


void UniformMeshVolumeCuda::VertexExtraction(
    UniformTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller::VertexExtraction(*this, tsdf_volume);
    timer.Stop();
    utility::PrintInfo("Extraction takes %f milliseconds\n", timer.GetDuration
    ());
}


void UniformMeshVolumeCuda::TriangleExtraction() {
    assert(device_ != nullptr);

    utility::Timer timer;
    timer.Start();

    UniformMeshVolumeCudaKernelCaller::TriangleExtraction(*this);

    timer.Stop();
    utility::PrintInfo("Triangulation takes %f milliseconds\n", timer
    .GetDuration());
}


void UniformMeshVolumeCuda::MarchingCubes(
    UniformTSDFVolumeCuda &tsdf_volume) {
    assert(device_ != nullptr && vertex_type_ != VertexTypeUnknown);

    mesh_.Reset();

    VertexAllocation(tsdf_volume);
    VertexExtraction(tsdf_volume);

    TriangleExtraction();

    if (vertex_type_ & VertexWithNormal) {
        mesh_.vertex_normals_.set_iterator(mesh_.vertices_.size());
    }
    if (vertex_type_ & VertexWithColor) {
        mesh_.vertex_colors_.set_iterator(mesh_.vertices_.size());
    }
}
} // cuda
} // open3d