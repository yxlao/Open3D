//
// Created by wei on 10/31/18.
//

#pragma once

#include "NormalShaderCuda.h"
#include "PhongShaderCuda.h"
#include "SimpleShaderCuda.h"
#include "SimpleBlackShaderCuda.h"

#include <Visualization/Shader/GeometryRenderer.h>

namespace open3d {

namespace glsl {

class TriangleMeshCudaRenderer : public GeometryRenderer {
public:
    ~TriangleMeshCudaRenderer() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    NormalShaderForTriangleMeshCuda normal_mesh_shader_;
    PhongShaderForTriangleMeshCuda phong_mesh_shader_;
    SimpleShaderForTriangleMeshCuda simple_mesh_shader_;
    SimpleBlackShaderForTriangleMeshCuda simpleblack_wireframe_shader_;
};

class PointCloudCudaRenderer : public GeometryRenderer {
public:
    ~PointCloudCudaRenderer() override = default;

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(std::shared_ptr<const Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    NormalShaderForTriangleMeshCuda normal_mesh_shader_;
    PhongShaderForTriangleMeshCuda phong_mesh_shader_;
    SimpleShaderForPointCloudCuda simple_mesh_shader_;
    SimpleBlackShaderForTriangleMeshCuda simpleblack_wireframe_shader_;
};
}

}


