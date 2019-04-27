//
// Created by wei on 4/15/19.
//

#include "BackgroundShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Visualization/Shader/Shader.h>
#include <InverseRendering/Visualization/Shader/Primitives.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool BackgroundShader::Compile() {
    if (!CompileShaders(BackgroundVertexShader,
                        nullptr,
                        BackgroundFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");
    tex_env_ = glGetUniformLocation(program_, "tex_env");

    return true;
}

void BackgroundShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool BackgroundShader::BindGeometry(const geometry::Geometry &geometry,
                                    const RenderOption &option,
                                    const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Create buffers and bind the geometry
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3i> triangles;
    if (!PrepareBinding(geometry, option, view, points, triangles)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    triangle_buffer_ = BindBuffer(triangles, GL_ELEMENT_ARRAY_BUFFER, option);
    bound_ = true;
    return true;
}

bool BackgroundShader::BindLighting(const geometry::Lighting &lighting,
                                    const visualization::RenderOption &option,
                                    const visualization::ViewControl &view) {
    auto ibl = (const geometry::IBLLighting &) lighting;
    ibl_ = ibl;
    return true;
}

bool BackgroundShader::RenderGeometry(const geometry::Geometry &geometry,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    glUseProgram(program_);

    /** 1. Set uniforms **/
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());

    /** 2. Set textures **/
    glUniform1i(tex_env_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ibl_.tex_env_buffer_);

    /** 3. Set up buffers **/
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

    glDrawElements(draw_arrays_mode_,
                   draw_arrays_size_,
                   GL_UNSIGNED_INT,
                   nullptr);
    glDisableVertexAttribArray(0);

    return true;
}

void BackgroundShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);

        bound_ = false;
    }
}

bool BackgroundShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3i> &triangles) {

    /** Prepare data **/
    LoadCube(points, triangles);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
