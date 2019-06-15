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

#include "UVForwardShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <AdvancedRendering/Geometry/TexturedTriangleMesh.h>
#include <AdvancedRendering/Visualization/Shader/Shader.h>
#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool UVForwardShader::Compile() {
    if (!CompileShaders(UVForwardVertexShader, nullptr,
                        UVForwardFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    M_ = glGetUniformLocation(program_, "M");
    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");

    texes_object_symbols_.resize(kNumObjectTextures);
    texes_object_symbols_[0] = glGetUniformLocation(program_, "tex_albedo");

    CheckGLState(GetShaderName() + ".Compile()");

    return true;
}

void UVForwardShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool UVForwardShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector2f> uvs;
    std::vector<Eigen::Vector3i> triangles;

    if (!PrepareBinding(geometry, option, view, points, uvs, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    vertex_uv_buffer_ = BindBuffer(uvs, GL_ARRAY_BUFFER, option);
    triangle_buffer_ = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);

    CheckGLState(GetShaderName() + ".BindGeometry()");

    auto mesh = (const geometry::TexturedTriangleMesh &)geometry;
    assert(mesh.texture_images_.size() >= kNumObjectTextures);
    texes_object_buffers_.resize(kNumObjectTextures);
    for (int i = 0; i < kNumObjectTextures; ++i) {
        texes_object_buffers_[i] =
                BindTexture2D(mesh.texture_images_[i], option);
    }

    CheckGLState(GetShaderName() + ".BindTexture()");

    bound_ = true;
    return true;
}

bool UVForwardShader::RenderGeometry(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    auto advanced_option = (const RenderOptionAdvanced &)option;

    if (advanced_option.render_to_fbo_) {
        glBindFramebuffer(GL_FRAMEBUFFER, advanced_option.fbo_forward_);
        glBindRenderbuffer(GL_RENDERBUFFER, advanced_option.rbo_forward_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               advanced_option.tex_forward_image_buffer_, 0);
        glClearTexImage(advanced_option.tex_forward_image_buffer_, 0,
                GL_RGB, GL_FLOAT, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D,
                               advanced_option.tex_forward_depth_buffer_, 0);
    }

    glUseProgram(program_);
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());

    /** Object buffers **/
    for (int i = 0; i < kNumObjectTextures; ++i) {
        glUniform1i(texes_object_symbols_[i], i);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, texes_object_buffers_[i]);
    }

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    if (advanced_option.render_to_fbo_) {
        std::vector<GLenum> draw_buffers;
        for (int i = 0; i < 1; ++i) {
            draw_buffers.emplace_back(GL_COLOR_ATTACHMENT0 + i);
        }
        glDrawBuffers(draw_buffers.size(), draw_buffers.data());
    }

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(draw_arrays_mode_, draw_arrays_size_, GL_UNSIGNED_INT,
                   nullptr);
//    glDrawArrays(GL_POINTS, 0, draw_arrays_size_);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    if (advanced_option.render_to_fbo_) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    CheckGLState(GetShaderName() + ".Render()");
    return true;
}

void UVForwardShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        for (int i = 0; i < kNumObjectTextures; ++i) {
            glDeleteTextures(1, &texes_object_buffers_[i]);
        }

        bound_ = false;
    }
}

bool UVForwardShader::PrepareRendering(const geometry::Geometry &geometry,
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
    glDepthFunc(GL_LEQUAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return true;
}

bool UVForwardShader::PrepareBinding(const geometry::Geometry &geometry,
                                     const RenderOption &option,
                                     const ViewControl &view,
                                     std::vector<Eigen::Vector3f> &points,
                                     std::vector<Eigen::Vector2f> &uvs,
                                     std::vector<Eigen::Vector3i> &triangles) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TexturedTriangleMesh) {
        PrintShaderWarning(
                "Rendering type is not geometry::TexturedTriangleMesh.");
        return false;
    }
    auto &mesh = (const geometry::TexturedTriangleMesh &)geometry;
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
