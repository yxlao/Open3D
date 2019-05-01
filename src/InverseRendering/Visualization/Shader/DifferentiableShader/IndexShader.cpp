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

#include "IndexShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <InverseRendering/Visualization/Shader/Shader.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool IndexShader::Compile() {
    std::cout << glGetError() << "\n";
    if (!CompileShaders(IndexVertexShader, nullptr, IndexFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");

    CheckGLState("IndexShader - Compile");

    return true;
}

void IndexShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool IndexShader::BindGeometry(const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector3i> triangles;

    if (!PrepareBinding(geometry, option, view, points, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    triangle_buffer_ = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);

    CheckGLState("IndexShader - BindGeometry");

    bound_ = true;
    return true;
}

bool IndexShader::RenderGeometry(const geometry::Geometry &geometry,
                                 const RenderOption &option,
                                 const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }


    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                          view.GetWindowWidth(), view.GetWindowHeight());
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_index_buffer_, 0);

    glUseProgram(program_);
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(draw_arrays_mode_, draw_arrays_size_, GL_UNSIGNED_INT,
                   nullptr);
    CheckGLState("IndexShader - Rendering Pass #1");
    /* Directly render it on previous layer for error check
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDrawArrays(GL_POINTS, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
     */

    /** Reuse depth buffer to occlude points, only clear color buffer **/
//    glClear(GL_COLOR_BUFFER_BIT);
//    glDrawArrays(GL_POINTS, 0, draw_arrays_size_);
//    CheckGLState("IndexShader - Rendering Pass #2");

    glDisableVertexAttribArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    /** Read the texture **/
    glBindTexture(GL_TEXTURE_2D, tex_index_buffer_);
    fbo_outputs_[0] = ReadTexture2D(
        view.GetWindowWidth(), view.GetWindowHeight(), 1, 4,
        GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_INT);

    /* Output indices for sanity check */
//    for (int u = 0; u < index_map_->width_; ++u) {
//        for (int v = 0; v < index_map_->height_; ++v) {
//            int* idx = geometry::PointerAt<int>(*index_map_, u, v);
//            if (*idx != 0) std::cout << "(" << u << ", " << v << ") " << *idx << "\n";
//        }
//    }

    CheckGLState("IndexShader - Render");
    return true;
}

void IndexShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        glDeleteBuffers(1, &tex_index_buffer_);
        bound_ = false;
    }
}

bool IndexShader::PrepareRendering(
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

bool IndexShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
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

    points.resize(mesh.vertices_.size());
    for (int i = 0; i < points.size(); ++i) {
        points[i] = mesh.vertices_[i].cast<float>();
    }
    triangles = mesh.triangles_;

    tex_index_buffer_ = CreateTexture2D(
        view.GetWindowWidth(), view.GetWindowHeight(),
        GL_LUMINANCE32UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_INT,
        false, option);
    fbo_outputs_.resize(1);

    CheckGLState("IndexShader - PrepareBinding");
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
