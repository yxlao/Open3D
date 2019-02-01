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

#include "NormalShaderCuda.h"

#include <Core/Geometry/PointCloud.h>
#include <Core/Geometry/TriangleMesh.h>
#include <Visualization/Shader/Shader.h>

namespace open3d {

namespace glsl {

bool NormalShaderCuda::Compile() {
    if (!CompileShaders(NormalVertexShader, NULL, NormalFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    return true;
}

void NormalShaderCuda::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool NormalShaderCuda::BindGeometry(const Geometry &geometry,
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
    if (!PrepareBinding(geometry, option, view)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    const cuda::TriangleMeshCuda &mesh =
        (const cuda::TriangleMeshCuda &)geometry;

    // Create buffers and bind the geometry
    RegisterResource(vertex_position_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_position_buffer_,
                     mesh.vertices_.device_->data(),
                     mesh.vertices_.size());

    RegisterResource(vertex_normal_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_normal_buffer_,
                     mesh.vertex_normals_.device_->data(),
                     mesh.vertex_normals_.size());

    RegisterResource(triangle_cuda_resource_,
                     GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_,
                     mesh.triangles_.device_->data(),
                     mesh.triangles_.size());

    bound_ = true;
    return true;
}

bool NormalShaderCuda::RenderGeometry(const Geometry &geometry,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    /* No vertex attrib array is required for ELEMENT_ARRAY_BUFFER */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_, draw_arrays_size_,
                   GL_UNSIGNED_INT, nullptr);

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);

    return true;
}

void NormalShaderCuda::UnbindGeometry() {
    if (bound_) {
        UnregisterResource(vertex_position_cuda_resource_,
                           vertex_position_buffer_);
        UnregisterResource(vertex_normal_cuda_resource_,
                           vertex_normal_buffer_);
        UnregisterResource(triangle_cuda_resource_,
                           triangle_buffer_);
        bound_ = false;
    }
}

bool NormalShaderForTriangleMeshCuda::PrepareRendering(const Geometry &geometry,
                                                       const RenderOption &option,
                                                       const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        Geometry::GeometryType::TriangleMeshCuda) {
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

bool NormalShaderForTriangleMeshCuda::PrepareBinding(const Geometry &geometry,
                                                     const RenderOption &option,
                                                     const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }

    const cuda::TriangleMeshCuda &mesh =
        (const cuda::TriangleMeshCuda &) geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

}    // namespace glsl

}    // namespace open3d
