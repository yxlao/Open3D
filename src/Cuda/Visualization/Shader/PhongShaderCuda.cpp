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

#include "PhongShaderCuda.h"
#include "CudaGLInterp.h"
#include <Open3D/Geometry/PointCloud.h>
#include <Cuda/Geometry/TriangleMeshCuda.h>
#include <Open3D/Visualization/Shader/Shader.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

namespace open3d {
namespace visualization {
namespace glsl {

bool PhongShaderCuda::Compile() {
    if (CompileShaders(PhongVertexShader, NULL, PhongFragmentShader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    light_position_world_ =
        glGetUniformLocation(program_, "light_position_world_4");
    light_color_ = glGetUniformLocation(program_, "light_color_4");
    light_diffuse_power_ = glGetUniformLocation(program_,
                                                "light_diffuse_power_4");
    light_specular_power_ = glGetUniformLocation(program_,
                                                 "light_specular_power_4");
    light_specular_shininess_ = glGetUniformLocation(program_,
                                                     "light_specular_shininess_4");
    light_ambient_ = glGetUniformLocation(program_, "light_ambient");
    return true;
}

void PhongShaderCuda::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool PhongShaderCuda::BindGeometry(const geometry::Geometry &geometry,
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
        (const cuda::TriangleMeshCuda &) geometry;

    // Create buffers and bind the geometry
    RegisterResource(vertex_position_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_position_buffer_,
                     mesh.vertices_.device_->data(),
                     mesh.vertices_.size());

    RegisterResource(vertex_normal_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_normal_buffer_,
                     mesh.vertex_normals_.device_->data(),
                     mesh.vertex_normals_.size());

    RegisterResource(vertex_color_cuda_resource_,
                     GL_ARRAY_BUFFER, vertex_color_buffer_,
                     mesh.vertex_colors_.device_->data(),
                     mesh.vertex_colors_.size());

    RegisterResource(triangle_cuda_resource_,
                     GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_,
                     mesh.triangles_.device_->data(),
                     mesh.triangles_.size());

    bound_ = true;
    return true;
}

bool PhongShaderCuda::RenderGeometry(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(light_position_world_, 1, GL_FALSE,
                       light_position_world_data_.data());
    glUniformMatrix4fv(light_color_, 1, GL_FALSE, light_color_data_.data());
    glUniform4fv(light_diffuse_power_, 1, light_diffuse_power_data_.data());
    glUniform4fv(light_specular_power_, 1, light_specular_power_data_.data());
    glUniform4fv(light_specular_shininess_, 1,
                 light_specular_shininess_data_.data());
    glUniform4fv(light_ambient_, 1, light_ambient_data_.data());

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_,
                   draw_arrays_size_,
                   GL_UNSIGNED_INT,
                   nullptr);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void PhongShaderCuda::UnbindGeometry() {
    if (bound_) {
        UnregisterResource(vertex_position_cuda_resource_,
                           vertex_position_buffer_);
        UnregisterResource(vertex_normal_cuda_resource_,
                           vertex_normal_buffer_);
        UnregisterResource(vertex_color_cuda_resource_,
                           vertex_color_buffer_);
        UnregisterResource(triangle_cuda_resource_,
                           triangle_buffer_);
        bound_ = false;
    }
}

void PhongShaderCuda::SetLighting(const ViewControl &view,
                                  const RenderOption &option) {
    const auto &box = view.GetBoundingBox();
    light_position_world_data_.setOnes();
    light_color_data_.setOnes();
    for (int i = 0; i < 4; i++) {
        light_position_world_data_.block<3, 1>(0, i) =
            box.GetCenter().cast<GLfloat>() + (float) box.GetSize() * (
                (float) option.light_position_relative_[i](0) * view.GetRight()
                    + (float) option.light_position_relative_[i](1)
                        * view.GetUp()
                    + (float) option.light_position_relative_[i](2)
                        * view.GetFront());
        light_color_data_.block<3, 1>(0, i) =
            option.light_color_[i].cast<GLfloat>();
    }
    if (option.light_on_) {
        light_diffuse_power_data_ = Eigen::Vector4d(
            option.light_diffuse_power_).cast<GLfloat>();
        light_specular_power_data_ = Eigen::Vector4d(
            option.light_specular_power_).cast<GLfloat>();
        light_specular_shininess_data_ = Eigen::Vector4d(
            option.light_specular_shininess_).cast<GLfloat>();
        light_ambient_data_.block<3, 1>(0, 0) =
            option.light_ambient_color_.cast<GLfloat>();
        light_ambient_data_(3) = 1.0f;
    } else {
        light_diffuse_power_data_ = GLHelper::GLVector4f::Zero();
        light_specular_power_data_ = GLHelper::GLVector4f::Zero();
        light_specular_shininess_data_ = GLHelper::GLVector4f::Ones();
        light_ambient_data_ = GLHelper::GLVector4f(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

bool PhongShaderForTriangleMeshCuda::PrepareRendering(const geometry::Geometry&geometry,
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
    SetLighting(view, option);
    return true;
}

bool PhongShaderForTriangleMeshCuda::PrepareBinding(const geometry::Geometry&geometry,
                                                    const RenderOption &option,
                                                    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMeshCuda) {
        PrintShaderWarning("Rendering type is not TriangleMeshCuda.");
        return false;
    }

    const cuda::TriangleMeshCuda &mesh =
        (const cuda::TriangleMeshCuda &) geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    if (!mesh.HasVertexNormals()) {
        PrintShaderWarning("Binding failed because mesh has no normals.");
        PrintShaderWarning("Call ComputeVertexNormals() before binding.");
        return false;
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

}    // namespace glsl
}    // visualization
}    // namespace open3d
