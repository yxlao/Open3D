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

#include "DifferentialShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <InverseRendering/Visualization/Shader/Shader.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool DifferentialShader::Compile() {
    std::cout << glGetError() << "\n";
    if (!CompileShaders(IBLNoTexVertexShader, nullptr, DifferentialFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    M_ = glGetUniformLocation(program_, "M");
    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");
    camera_position_ = glGetUniformLocation(program_, "camera_position");
    viewport_ = glGetUniformLocation(program_, "viewport");

    texes_env_.resize(kNumEnvTextures);
    texes_env_[0] = glGetUniformLocation(program_, "tex_env_diffuse");
    texes_env_[1] = glGetUniformLocation(program_, "tex_env_specular");
    texes_env_[2] = glGetUniformLocation(program_, "tex_lut_specular");
    tex_target_image_ = glGetUniformLocation(program_, "tex_target_image");

    CheckGLState("SceneDifferentialShader - Compile");

    return true;
}

void DifferentialShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool DifferentialShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector3f> colors;
    std::vector<Eigen::Vector3f> materials;
    std::vector<Eigen::Vector3i> triangles;

    if (!PrepareBinding(geometry, option, view,
                        points, normals, colors, materials, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    vertex_normal_buffer_ = BindBuffer(normals, GL_ARRAY_BUFFER, option);
    vertex_color_buffer_ = BindBuffer(colors, GL_ARRAY_BUFFER, option);
    vertex_material_buffer_ = BindBuffer(materials, GL_ARRAY_BUFFER, option);

    triangle_buffer_ = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);

    CheckGLState("SceneDifferentialShader - BindGeometry");

    bound_ = true;
    return true;
}

bool DifferentialShader::BindTextures(const std::vector<open3d::geometry::Image> &textures,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    assert(textures.size() == 1);
    tex_target_img_buffer_ = BindTexture2D(textures[0], option);

    CheckGLState("DifferentialShader - BindTexture");
    return true;
}

bool DifferentialShader::BindLighting(const geometry::Lighting &lighting,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    auto ibl = (const geometry::IBLLighting &) lighting;

    tex_env_buffers_.resize(kNumEnvTextures);
    tex_env_buffers_[0] = ibl.tex_env_diffuse_buffer_;
    tex_env_buffers_[1] = ibl.tex_env_specular_buffer_;
    tex_env_buffers_[2] = ibl.tex_lut_specular_buffer_;

    for (int i = 0; i < kNumEnvTextures; ++i) {
        std::cout << "tex_obejct_buffer: " << tex_env_buffers_[i] << "\n";
    }
    return true;
}

bool DifferentialShader::RenderGeometry(const geometry::Geometry &geometry,
                               const RenderOption &option,
                               const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    CheckGLState("SceneDifferentialShader - Before Render");

    if (! is_debug_) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, rbo_);
        for (int i = 0; i < kNumOutputTextures; ++i) {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                                   GL_TEXTURE_2D, tex_fbo_buffers_[i], 0);
        }
    }

    glUseProgram(program_);
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());
    glUniform3fv(camera_position_, 1, view.GetEye().data());
    glUniform2fv(viewport_, 1, Eigen::Vector2f(
        view.GetWindowWidth(),view.GetWindowHeight()).data());

    /** Diffuse environment **/
    glUniform1i(texes_env_[0], 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env_buffers_[0]);

    /** Prefiltered specular **/
    glUniform1i(texes_env_[1], 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env_buffers_[1]);

    /** Pre-integrated BRDF LUT **/
    glUniform1i(texes_env_[2], 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_env_buffers_[2]);

    glUniform1i(tex_target_image_, 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_target_img_buffer_);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(3);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_material_buffer_);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    if (! is_debug_) {
        std::vector<GLenum> draw_buffers;
        for (int i = 0; i < kNumOutputTextures; ++i) {
            draw_buffers.emplace_back(GL_COLOR_ATTACHMENT0 + i);
        }
        glDrawBuffers(draw_buffers.size(), draw_buffers.data());
    }

    glDrawElements(draw_arrays_mode_, draw_arrays_size_, GL_UNSIGNED_INT, nullptr);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);

    if (! is_debug_) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        for (int i = 0; i < tex_fbo_buffers_.size(); ++i) {
            glBindTexture(GL_TEXTURE_2D, tex_fbo_buffers_[i]);
            fbo_outputs_[i] = ReadTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(), 3, 4,
                GL_RGB, GL_FLOAT);
        }
    }

    CheckGLState("SceneDifferentialShader - Render");
    return true;
}

void DifferentialShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        glDeleteBuffers(1, &vertex_material_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        if (! is_debug_) {
//            for (int i = 0; i < kNumEnvTextures; ++i) {
//                glDeleteTextures(1, &tex_env_buffers_[i]);
//            }
//            for (int i = 0; i < kNumOutputTextures; ++i) {
//                glDeleteTextures(1, &tex_fbo_buffers_[i]);
//            }

//            glDeleteFramebuffers(1, &fbo_);
//            glDeleteRenderbuffers(1, &rbo_);
        }

        bound_ = false;
    }
}

bool DifferentialShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
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

bool DifferentialShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3f> &normals,
    std::vector<Eigen::Vector3f> &colors,
    std::vector<Eigen::Vector3f> &materials,
    std::vector<Eigen::Vector3i> &triangles) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning(
            "Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    auto &mesh = (const geometry::TriangleMeshExtended &) geometry;
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
    colors.resize(mesh.vertex_colors_.size());
    for (int i = 0; i < colors.size(); ++i) {
        colors[i] = mesh.vertex_colors_[i].cast<float>();
    }
    materials.resize(mesh.vertex_materials_.size());
    for (int i = 0; i < colors.size(); ++i) {
        materials[i] = mesh.vertex_materials_[i].cast<float>();
    }
    triangles = mesh.triangles_;

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);

    if (! is_debug_) {
        tex_fbo_buffers_.resize(kNumOutputTextures);
        fbo_outputs_.resize(kNumOutputTextures);
        for (int i = 0; i < kNumOutputTextures; ++i) {
            tex_fbo_buffers_[i] = CreateTexture2D(
                view.GetWindowWidth(), view.GetWindowHeight(),
                GL_RGB16F, GL_RGB, GL_FLOAT, false, option);
        }

        glGenFramebuffers(1, &fbo_);
        glGenRenderbuffers(1, &rbo_);

        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              view.GetWindowWidth(), view.GetWindowHeight());
    }
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
