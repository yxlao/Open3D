//
// Created by wei on 4/15/19.
//

#include "PreIntegrateLUTSpecularShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <InverseRendering/Visualization/Shader/Shader.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Visualization/Shader/Primitives.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool PreIntegrateLUTSpecularShader::Compile() {
    if (!CompileShaders(PreIntegrateLUTVertexShader,
                        nullptr,
                        PreIntegrateLUTFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    return true;
}

void PreIntegrateLUTSpecularShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool PreIntegrateLUTSpecularShader::BindGeometry(const geometry::Geometry &geometry,
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

bool PreIntegrateLUTSpecularShader::BindLighting(const geometry::Lighting &lighting,
                                                 const visualization::RenderOption &option,
                                                 const visualization::ViewControl &view) {
    return true;
}

bool PreIntegrateLUTSpecularShader::RenderGeometry(const geometry::Geometry &geometry,
                                                   const RenderOption &option,
                                                   const ViewControl &view) {
    if (!PrepareRendering(geometry, option, view)) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    /** 0. Setup framebuffers **/
    GLuint fbo, rbo;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                          kTextureSize, kTextureSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_lut_specular_buffer_, 0);
    glViewport(0, 0, kTextureSize, kTextureSize);


    /** Setup programs and unchanged uniforms **/
    glUseProgram(program_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);

    glViewport(0, 0, view.GetWindowWidth(), view.GetWindowHeight());

    return true;
}

void PreIntegrateLUTSpecularShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);

        bound_ = false;
    }
}

bool PreIntegrateLUTSpecularShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {

    /** Additional states **/
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);

    return true;
}

bool PreIntegrateLUTSpecularShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector2f> &uvs) {

    /** Prepare data **/
    LoadQuad(points, uvs);

    /** Prepare target texture **/
    tex_lut_specular_buffer_ = CreateTexture2D(
        kTextureSize, kTextureSize,
        GL_RGB16F, GL_RG, GL_FLOAT, false, option);

    draw_arrays_mode_ = GL_TRIANGLE_STRIP;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
