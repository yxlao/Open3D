//
// Created by wei on 4/15/19.
//

#include "BackgroundShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <Material/Visualization/Shader/Shader.h>
#include <Material/Physics/TriangleMeshWithTex.h>
#include <Material/Physics/Cube.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool BackgroundShader::Compile() {
    if (! CompileShaders(BackgroundVertexShader, nullptr, BackgroundFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    vertex_position_ = glGetAttribLocation(program_, "vertex_position");

    V_               = glGetUniformLocation(program_, "V");
    P_               = glGetUniformLocation(program_, "P");

    tex_cubemap_     = glGetUniformLocation(program_, "tex_cubemap");

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
    PrepareBinding(geometry, option, view, points);
    vertex_position_buffer_ = BindBuffer(points, GL_ARRAY_BUFFER, option);

    bound_ = true;
    return true;
}

bool BackgroundShader::BindLighting(const physics::Lighting &lighting,
                                      const visualization::RenderOption &option,
                                      const visualization::ViewControl &view) {
    auto ibl = (const physics::IBLLighting &) lighting;
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
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(P_, 1, GL_FALSE, view.GetProjectionMatrix().data());

    glUniform1i(tex_cubemap_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ibl_.tex_cubemap_buffer_);

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);

    return true;
}

void BackgroundShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);

        bound_ = false;
    }
}

bool BackgroundShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {

    /** Additional states **/
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
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    return true;
}

bool BackgroundShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points) {

    /** Prepare data **/
    points.resize(physics::kCubeVertices.size() / 3);
    for (int i = 0; i < points.size(); i += 3) {
        points[i] = Eigen::Vector3f(physics::kCubeVertices[i + 0],
                                    physics::kCubeVertices[i + 1],
                                    physics::kCubeVertices[i + 2]);
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(36);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
