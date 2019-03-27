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

#include "SimpleBlackShaderCuda.h"
#include "CudaGLInterp.h"

#include <Open3D/Geometry/PointCloud.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Open3D/Visualization/Shader/Shader.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

namespace open3d {
namespace visualization {
namespace glsl {

bool SimpleBlackShaderCuda::Compile() {
    if (CompileShaders(SimpleBlackVertexShader, NULL,
                       SimpleBlackFragmentShader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleBlackShaderCuda::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleBlackShaderCuda::BindGeometry(const geometry::Geometry &geometry,
                                         const RenderOption &option,
                                         const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    if (PrepareBinding(geometry, option, view) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    const cuda::TriangleMeshCuda &mesh =
        (const cuda::TriangleMeshCuda &) geometry;
    RegisterResource(vertex_position_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_position_buffer_,
                     mesh.vertices_.device_->data(),
                     mesh.vertices_.size());

    RegisterResource(triangle_cuda_resource,
                     GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_,
                     mesh.triangles_.device_->data(),
                     mesh.triangles_.size());

    bound_ = true;
    return true;
}

bool SimpleBlackShaderCuda::RenderGeometry(const geometry::Geometry &geometry,
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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_,
                   draw_arrays_size_,
                   GL_UNSIGNED_INT,
                   nullptr);
    glDisableVertexAttribArray(vertex_position_);
    return true;
}

void SimpleBlackShaderCuda::UnbindGeometry() {
    if (bound_) {
        UnregisterResource(vertex_position_cuda_resource_,
                           vertex_position_buffer_);
        UnregisterResource(triangle_cuda_resource,
                           triangle_buffer_);
        bound_ = false;
    }
}

bool SimpleBlackShaderForTriangleMeshCuda::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_POLYGON_OFFSET_FILL);

    return true;
}

bool SimpleBlackShaderForTriangleMeshCuda::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }

    const cuda::TriangleMeshCuda &mesh =
        (const cuda::TriangleMeshCuda &) geometry;
    if (mesh.HasTriangles() == false) {
        PrintShaderWarning("Binding failed with empty TriangleMeshCuda.");
        return false;
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);

    return true;
}

}
}
}    // namespace open3d
