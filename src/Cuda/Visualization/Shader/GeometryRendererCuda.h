//
// Created by wei on 10/31/18.
//

#pragma once

#include "NormalShaderCuda.h"
#include <Visualization/Shader/GeometryRenderer.h>

namespace open3d {

namespace glsl {

/**
 * Replacement of BindData - direct transfer data to graphics part
 */
template<typename T>
bool BindCudaGraphicsResource(GLuint &opengl_buffer,
                              cudaGraphicsResource_t &cuda_graphics_resource,
                              T *cuda_vector,
                              size_t cuda_vector_size) {

    CheckCuda(cudaGraphicsMapResources(1, &cuda_graphics_resource));
    void *mapped_ptr;
    size_t mapped_size;
    CheckCuda(cudaGraphicsResourceGetMappedPointer(
        &mapped_ptr, &mapped_size, cuda_graphics_resource));
    CheckCuda(cudaMemcpy(mapped_ptr, cuda_vector, sizeof(T) * cuda_vector_size,
                         cudaMemcpyDeviceToDevice));
    CheckCuda(cudaGraphicsUnmapResources(1, &cuda_graphics_resource, nullptr));

    return true;
}

class TriangleMeshCudaRenderer : public GeometryRenderer {
public:
    ~TriangleMeshCudaRenderer() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    NormalShaderForTriangleMeshCuda normal_mesh_shader_;
};
}

}


