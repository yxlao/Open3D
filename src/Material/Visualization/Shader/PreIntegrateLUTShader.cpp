//
// Created by wei on 4/15/19.
//

#include "PreIntegrateLUTShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <Material/Visualization/Shader/Shader.h>
#include <Material/Physics/TriangleMeshWithTex.h>
#include <Material/Physics/Primitives.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool PreIntegrateLUTShader::Compile() {
    if (! CompileShaders(PreIntegrateLUTVertexShader, nullptr, PreIntegrateLUTFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_uv_       = glGetAttribLocation(program_, "vertex_uv");

    return true;
}

void PreIntegrateLUTShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool PreIntegrateLUTShader::BindGeometry(const geometry::Geometry &geometry,
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
    vertex_uv_buffer_       = BindBuffer(uvs, GL_ARRAY_BUFFER, option);
    bound_ = true;
    return true;
}

bool PreIntegrateLUTShader::BindLighting(const physics::Lighting &lighting,
                                         const visualization::RenderOption &option,
                                         const visualization::ViewControl &view) {
    return true;
}

bool PreIntegrateLUTShader::RenderGeometry(const geometry::Geometry &geometry,
                                           const RenderOption &option,
                                           const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    /** Setup framebuffers **/
    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                          kTextureSize, kTextureSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    /** Setup programs and unchanged uniforms **/
    glUseProgram(program_);

    glViewport(0, 0, kTextureSize, kTextureSize);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_brdf_lut_buffer_, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_uv_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_uv_);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);

    glViewport(0, 0, view.GetWindowWidth(), view.GetWindowHeight());

    return true;
}

void PreIntegrateLUTShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);

        bound_ = false;
    }
}

bool PreIntegrateLUTShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {

    /** Additional states **/
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    return true;
}

bool PreIntegrateLUTShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector2f> &uvs) {

    /** Prepare data **/
    points.resize(physics::kQuadVertices.size() / 3);
    uvs.resize(physics::kQuadUVs.size() / 2);
    for (int i = 0; i < points.size(); ++i) {
        points[i] = Eigen::Vector3f(physics::kQuadVertices[i * 3 + 0],
                                    physics::kQuadVertices[i * 3 + 1],
                                    physics::kQuadVertices[i * 3 + 2]);
        uvs[i] = Eigen::Vector2f(physics::kQuadUVs[i * 2 + 0],
                                 physics::kQuadUVs[i * 2 + 1]);
    }

    /** Prepare target texture **/
    glGenTextures(1, &tex_brdf_lut_buffer_);
    glBindTexture(GL_TEXTURE_2D, tex_brdf_lut_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB16F, kTextureSize, kTextureSize, 0, GL_RG, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    draw_arrays_mode_ = GL_TRIANGLE_STRIP;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
