// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "SimpleShaderCuda.h"
#include "CudaGLInterp.h"
#include <Cuda/Common/UtilsCuda.h>

#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Cuda/Geometry/PointCloudCuda.h>

#include <Open3D/Visualization/Shader/Shader.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

namespace open3d {
namespace visualization {
namespace glsl {

bool SimpleShaderCuda::Compile() {
    if (CompileShaders(SimpleVertexShader, NULL,
                       SimpleFragmentShader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleShaderCuda::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleShaderCuda::BindGeometry(const geometry::Geometry &geometry,
                                    const RenderOption &option,
                                    const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    cuda::Vector3f *vertices, *colors;
    cuda::Vector3i *triangles;
    int vertex_size, triangle_size;
    if (!PrepareBinding(geometry, option, view,
                        vertices, colors, triangles,
                        vertex_size, triangle_size)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    auto &pcl = (const cuda::PointCloudCuda &) geometry;
    RegisterResource(vertex_position_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_position_buffer_,
                     vertices, vertex_size);
    RegisterResource(vertex_color_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_color_buffer_,
                     colors, vertex_size);

    if (geometry.GetGeometryType() ==
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        RegisterResource(triangle_cuda_resource_,
                         GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_,
                         triangles, triangle_size);
    } else {
        triangle_cuda_resource_ = nullptr;
    }

    bound_ = true;
    return true;
}

bool SimpleShaderCuda::RenderGeometry(const geometry::Geometry &geometry,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    if (geometry.GetGeometryType() ==
        geometry::Geometry::GeometryType::PointCloudCuda) {
        glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    } else if (geometry.GetGeometryType() ==
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        glBindBuffer(GL_ARRAY_BUFFER, triangle_buffer_);
        glDrawElements(draw_arrays_mode_, draw_arrays_size_,
                       GL_UNSIGNED_INT,
                       NULL);
    } else {
        PrintShaderWarning("Rendering type is unknown.");
        return false;
    }

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void SimpleShaderCuda::UnbindGeometry() {
    if (bound_) {
        UnregisterResource(vertex_position_cuda_resource_,
                           vertex_position_buffer_);
        UnregisterResource(vertex_color_cuda_resource_,
                           vertex_color_buffer_);

        if (triangle_cuda_resource_) {
            UnregisterResource(triangle_cuda_resource_,
                               triangle_buffer_);
        }
        bound_ = false;
    }
}

/****** SimpleShaderForPointCloudCuda ******/
bool SimpleShaderForPointCloudCuda::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloudCuda) {
        PrintShaderWarning("Rendering type is not PointCloudCuda.");
        return false;
    }

    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    return true;
}

bool SimpleShaderForPointCloudCuda::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    cuda::Vector3f *&vertices,
    cuda::Vector3f *&colors,
    cuda::Vector3i *&triangles,
    int &vertex_size,
    int &triangle_size) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloudCuda) {
        PrintShaderWarning("Rendering type is not PointCloudCuda.");
        return false;
    }

    auto &pcl = (const cuda::PointCloudCuda &) geometry;
    if (!pcl.HasPoints()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }

    vertices = pcl.points_.device_->data();
    colors = pcl.colors_.device_->data();
    triangles = nullptr;

    vertex_size = pcl.points_.size();
    triangle_size = 0;

    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(vertex_size);
    return true;
}

/****** SimpleShaderForTriangleMeshCuda ******/
bool SimpleShaderForTriangleMeshCuda::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool SimpleShaderForTriangleMeshCuda::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    cuda::Vector3f *&vertices,
    cuda::Vector3f *&colors,
    cuda::Vector3i *&triangles,
    int &vertex_size,
    int &triangle_size) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }

    auto &mesh = (const cuda::TriangleMeshCuda &) geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }

    vertices = mesh.vertices_.device_->data();
    colors = mesh.vertex_colors_.device_->data();
    triangles = mesh.triangles_.device_->data();

    vertex_size = mesh.vertices_.size();
    triangle_size = mesh.triangles_.size();

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangle_size * 3);
    return true;
}

}    // namespace open3d::glsl
}
}    // namespace open3d
