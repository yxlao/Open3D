//
// Created by wei on 4/15/19.
//

#include "SimpleTextureShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <AdvancedRendering/Visualization/Shader/Shader.h>
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>
#include <AdvancedRendering/Visualization/Utility/Primitives.h>
#include <AdvancedRendering/Visualization/Visualizer/RenderOptionAdvanced.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool SimpleTextureShader::Compile() {
    if (!CompileShaders(SimpleTextureVertexShader,
                        nullptr,
                        SimpleTextureFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    texture_symbol_ = glGetUniformLocation(program_, "texture_vis");

    return true;
}

void SimpleTextureShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleTextureShader::BindGeometry(
    const geometry::Geometry &geometry,
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
    std::vector<Eigen::Vector2f> uvs;
    if (!PrepareBinding(geometry, option, view, points, uvs)) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);
    vertex_uv_buffer_ = BindBuffer(uvs, GL_ARRAY_BUFFER, option);
    bound_ = true;
    return true;
}

bool SimpleTextureShader::RenderGeometry(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    auto advanced_option = (const RenderOptionAdvanced &) option;

    /** Setup programs and unchanged uniforms **/
    glUseProgram(program_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUniform1i(texture_symbol_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, advanced_option.tex_visualize_buffer_);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glDisableVertexAttribArray(0);

    return true;
}

void SimpleTextureShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);

        bound_ = false;
    }
}

bool SimpleTextureShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {

    /** Additional states **/
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    return true;
}

bool SimpleTextureShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector2f> &uvs) {

    /** Prepare data **/
    LoadQuad(points, uvs);

    draw_arrays_mode_ = GL_TRIANGLE_STRIP;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
