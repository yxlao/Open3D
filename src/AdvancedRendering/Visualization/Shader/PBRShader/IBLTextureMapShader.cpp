//
// Created by wei on 4/13/19.
//
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

#include "IBLTextureMapShader.h"

#include <Open3D/Geometry/TriangleMesh.h>

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/Visualization/Shader/Shader.h>
#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool IBLTextureMapShader::Compile() {
    if (!CompileShaders(IBLTexMapVertexShader,
                        nullptr,
                        IBLTexMapFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    M_ = glGetUniformLocation(program_, "M");
    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");
    camera_position_ = glGetUniformLocation(program_, "camera_position");

    tex_object_symbols_.resize(kNumObjectTextures);
    tex_object_symbols_[0] = glGetUniformLocation(program_, "tex_albedo");
    tex_object_symbols_[1] = glGetUniformLocation(program_, "tex_normal");
    tex_object_symbols_[2] = glGetUniformLocation(program_, "tex_roughness");
    tex_object_symbols_[3] = glGetUniformLocation(program_, "tex_metallic");
    tex_object_symbols_[4] = glGetUniformLocation(program_, "tex_ao");

    tex_env_symbols_.resize(kNumEnvTextures);
    tex_env_symbols_[0] = glGetUniformLocation(program_, "tex_env_diffuse");
    tex_env_symbols_[1] = glGetUniformLocation(program_, "tex_env_specular");
    tex_env_symbols_[2] = glGetUniformLocation(program_, "tex_lut_specular");

    CheckGLState(GetShaderName() + ".Render");

    return true;
}

void IBLTextureMapShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool IBLTextureMapShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector2f> uvs;
    std::vector<Eigen::Vector3i> triangles;

    if (!PrepareBinding(geometry, option, view,
                        points, normals, uvs, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    vertex_normal_buffer_ = BindBuffer(normals, GL_ARRAY_BUFFER, option);
    vertex_uv_buffer_ = BindBuffer(uvs, GL_ARRAY_BUFFER, option);
    triangle_buffer_ = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);

    CheckGLState(GetShaderName() + ".BindGeometry");

    auto mesh = (const geometry::TexturedTriangleMesh &) geometry;
    assert(mesh.texture_images_.size() >= kNumObjectTextures);
    tex_object_buffers_.resize(mesh.texture_images_.size());
    for (int i = 0; i < mesh.texture_images_.size(); ++i) {
        tex_object_buffers_[i] = BindTexture2D(
            mesh.texture_images_[i], option);
    }

    CheckGLState(GetShaderName() + ".BindTexture");
    bound_ = true;
    return true;
}

bool IBLTextureMapShader::RenderGeometry(const geometry::Geometry &geometry,
                                         const RenderOption &option,
                                         const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    auto &lighting_option = (const RenderOptionAdvanced &) option;

    std::vector<GLuint> tex_env_buffers;
    tex_env_buffers.resize(kNumEnvTextures);
    tex_env_buffers[0] = lighting_option.tex_env_diffuse_buffer_;
    tex_env_buffers[1] = lighting_option.tex_env_specular_buffer_;
    tex_env_buffers[2] = lighting_option.tex_lut_specular_buffer_;

    glUseProgram(program_);
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());
    glUniform3fv(camera_position_, 1, (const GLfloat *) view.GetEye().data());

    /** Diffuse environment **/
    glUniform1i(tex_env_symbols_[0], 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env_buffers[0]);

    /** Prefiltered specular **/
    glUniform1i(tex_env_symbols_[1], 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env_buffers[1]);

    /** Pre-integrated BRDF LUT **/
    glUniform1i(tex_env_symbols_[2], 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_env_buffers[2]);

    /** Object buffers **/
    for (int i = 0; i < kNumObjectTextures; ++i) {
        glUniform1i(tex_object_symbols_[i], i + kNumEnvTextures);
        glActiveTexture(GL_TEXTURE0 + i + kNumEnvTextures);
        glBindTexture(GL_TEXTURE_2D, tex_object_buffers_[i]);
    }

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_, draw_arrays_size_, GL_UNSIGNED_INT,
                   nullptr);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);

    CheckGLState(GetShaderName() + ".Render");
    return true;
}

void IBLTextureMapShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        for (int i = 0; i < kNumObjectTextures; ++i) {
            glDeleteTextures(1, &tex_object_buffers_[i]);
        }

        bound_ = false;
    }
}

bool IBLTextureMapShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        PrintShaderWarning(
            "Rendering type is not geometry::TexturedTriangleMesh.");
        return false;
    }
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL); /** For the environment **/
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return true;
}

bool IBLTextureMapShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3f> &normals,
    std::vector<Eigen::Vector2f> &uvs,
    std::vector<Eigen::Vector3i> &triangles) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        PrintShaderWarning(
            "Rendering type is not geometry::TexturedTriangleMesh.");
        return false;
    }
    auto &mesh = (const geometry::TexturedTriangleMesh &) geometry;
    if (!mesh.HasTriangles()) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    if (!mesh.HasVertexNormals()) {
        PrintShaderWarning("Binding failed because mesh has no normals.");
        return false;
    }

    points.resize(mesh.vertices_.size());
    for (int i = 0; i < points.size(); ++i) {
        points[i] = mesh.vertices_[i].cast<float>();
    }
    normals.resize(mesh.vertex_normals_.size());
    for (int i = 0; i < normals.size(); ++i) {
        normals[i] = mesh.vertex_normals_[i].cast<float>();
    }
    uvs.resize(mesh.vertex_uvs_.size());
    for (int i = 0; i < uvs.size(); ++i) {
        uvs[i] = mesh.vertex_uvs_[i].cast<float>();
    }
    triangles = mesh.triangles_;

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
