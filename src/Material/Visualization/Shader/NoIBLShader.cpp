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

#include "NoIBLShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <Material/Visualization/Shader/Shader.h>
#include <Material/Physics/TriangleMeshPhysics.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool NoIBLShader::Compile() {
    GLenum err = glGetError();
    std::cout << "Capture error before compile " << err << std::endl;

    if (! CompileShaders(NoIBLVertexShader, NULL, NoIBLFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_   = glGetAttribLocation(program_, "vertex_normal");
    vertex_uv_       = glGetAttribLocation(program_, "vertex_uv");

    M_               = glGetUniformLocation(program_, "M");
    V_               = glGetUniformLocation(program_, "V");
    P_               = glGetUniformLocation(program_, "P");

    err = glGetError();
    std::cout << "GetUniform " << err << std::endl;

    tex_albedo_      = glGetUniformLocation(program_, "tex_albedo");
    tex_normal_      = glGetUniformLocation(program_, "tex_normal");
    tex_metallic_    = glGetUniformLocation(program_, "tex_metallic");
    tex_roughness_   = glGetUniformLocation(program_, "tex_roughness");
    tex_ao_          = glGetUniformLocation(program_, "tex_ao");

    light_positions_ = glGetUniformLocation(program_, "light_positions");
    light_colors_    = glGetUniformLocation(program_, "light_colors");
    camera_position_ = glGetUniformLocation(program_, "camera_position");

    return true;
}

void NoIBLShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool NoIBLShader::BindGeometry(const geometry::Geometry &geometry,
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

    if (!PrepareBinding(geometry, option, view, points, normals, uvs, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    vertex_normal_buffer_   = BindBuffer(normals, GL_ARRAY_BUFFER, option);
    vertex_uv_buffer_       = BindBuffer(uvs, GL_ARRAY_BUFFER, option);
    triangle_buffer_        = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);
    GLenum err = glGetError();
    std::cout << "BindGeometry " << err << std::endl;

    bound_ = true;
    return true;
}

bool NoIBLShader::BindTextures(const std::vector<geometry::Image> &textures,
                                  const RenderOption& option,
                                  const ViewControl &view) {
    assert(textures.size() == 5);

    tex_albedo_id_    = BindTexture(textures[0], option);
    tex_normal_id_    = BindTexture(textures[1], option);
    tex_metallic_id_  = BindTexture(textures[2], option);
    tex_roughness_id_ = BindTexture(textures[3], option);
    tex_ao_id_        = BindTexture(textures[4], option);
    GLenum err = glGetError();
    std::cout << "BindTexture " << err << std::endl;

    return true;
}

bool NoIBLShader::BindLighting(const physics::Lighting &lighting,
                                  const visualization::RenderOption &option,
                                  const visualization::ViewControl &view) {
    auto spot_lighting = (const physics::SpotLighting &) lighting;

    light_positions_data_ = spot_lighting.light_positions_;
    light_colors_data_    = spot_lighting.light_colors_;
}

bool NoIBLShader::RenderGeometry(const geometry::Geometry &geometry,
                                 const RenderOption &option,
                                 const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    glUseProgram(program_);
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());
    GLenum err = glGetError();
    std::cout << "Uniform " << err << std::endl;

    glUniform1i(tex_albedo_,    0);
    glUniform1i(tex_normal_,    1);
    glUniform1i(tex_metallic_,  2);
    glUniform1i(tex_roughness_, 3);
    glUniform1i(tex_ao_,        4);
    err = glGetError();
    std::cout << "tex " << err << std::endl;

    glUniform3fv(camera_position_, 1, view.GetEye().data());
    glUniform3fv(light_positions_, light_positions_data_.size(),
                 (const GLfloat*) light_positions_data_.data());
    glUniform3fv(light_colors_, light_colors_data_.size(),
                 (const GLfloat*) light_colors_data_.data());
    err = glGetError();
    std::cout << "bind uniform " << err << std::endl;

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_albedo_id_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex_normal_id_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, tex_metallic_id_);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, tex_roughness_id_);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, tex_ao_id_);
    err = glGetError();
    std::cout << "active and bind tex " << err << std::endl;

    glEnableVertexAttribArray(vertex_position_);
    err = glGetError();
    std::cout << "Position enable" << err << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    err = glGetError();
    std::cout << "Position bind" << err << std::endl;
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    err = glGetError();
    std::cout << "Position attrib" << err << std::endl;

    glEnableVertexAttribArray(vertex_normal_);
    err = glGetError();
    std::cout << "Normal enable" << err << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    err = glGetError();
    std::cout << "Normal bind" << err << std::endl;
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    err = glGetError();
    std::cout << "Normal attrib" << err << std::endl;

    glEnableVertexAttribArray(vertex_uv_);
    err = glGetError();
    std::cout << "uv enable" << err << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    err = glGetError();
    std::cout << "uv bind" << err << std::endl;
    glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    err = glGetError();
    std::cout << "uv attrib" << err << std::endl;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_,
                   draw_arrays_size_,
                   GL_UNSIGNED_INT,
                   nullptr);
    err = glGetError();
    std::cout << "Draw " << err << std::endl;

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    glDisableVertexAttribArray(vertex_uv_);
    err = glGetError();
    std::cout << "Disable " << err << std::endl;

    return true;
}

void NoIBLShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        glDeleteTextures(1, &tex_albedo_id_);
        glDeleteTextures(1, &tex_normal_id_);
        glDeleteTextures(1, &tex_metallic_id_);
        glDeleteTextures(1, &tex_roughness_id_);
        glDeleteTextures(1, &tex_ao_id_);

        bound_ = false;
    }
}

bool NoIBLShader::PrepareRendering(
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
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    return true;
}

bool NoIBLShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3f> &normals,
    std::vector<Eigen::Vector2f> &uvs,
    std::vector<Eigen::Vector3i> &triangles) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning(
            "Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    auto &mesh = (const geometry::TriangleMeshPhysics &) geometry;
    if (! mesh.HasTriangles()) {
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
    uvs = mesh.vertex_uvs_;
    triangles = mesh.triangles_;

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
