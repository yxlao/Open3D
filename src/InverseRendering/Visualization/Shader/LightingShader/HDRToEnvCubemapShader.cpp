//
// Created by wei on 4/15/19.
//

#include "HDRToEnvCubemapShader.h"

#include <Open3D/Geometry/TriangleMesh.h>
#include <Open3D/Visualization/Utility/ColorMap.h>

#include <InverseRendering/Visualization/Shader/Shader.h>
#include <InverseRendering/Geometry/TriangleMeshExtended.h>
#include <InverseRendering/Visualization/Shader/Primitives.h>

namespace open3d {
namespace visualization {

namespace glsl {

bool HDRToEnvCubemapShader::Compile() {
    if (!CompileShaders(SimpleVertexShader,
                        nullptr,
                        HDRToCubemapFragmentShader)) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }

    V_ = glGetUniformLocation(program_, "V");
    P_ = glGetUniformLocation(program_, "P");

    tex_hdr_ = glGetUniformLocation(program_, "tex_hdr");
    return true;
}

void HDRToEnvCubemapShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool HDRToEnvCubemapShader::BindGeometry(const geometry::Geometry &geometry,
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

bool HDRToEnvCubemapShader::BindLighting(const geometry::Lighting &lighting,
                                         const visualization::RenderOption &option,
                                         const visualization::ViewControl &view) {
    auto ibl = (const geometry::IBLLighting &) lighting;
    tex_hdr_buffer_ = ibl.tex_hdr_buffer_;

    return true;
}

bool HDRToEnvCubemapShader::RenderGeometry(const geometry::Geometry &geometry,
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
                          kCubemapSize, kCubemapSize);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbo);
    glViewport(0, 0, kCubemapSize, kCubemapSize);

    /** 1. Setup programs and constant uniforms **/
    glUseProgram(program_);
    glUniformMatrix4fv(P_, 1, GL_FALSE, projection_.data());

    /** 2. Setup constant textures **/
    glUniform1i(tex_hdr_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_hdr_buffer_);

    /** 3. Setup varying viewpoints and rendering target **/
    for (int i = 0; i < 6; ++i) {
        glUniformMatrix4fv(V_, 1, GL_FALSE, views_[i].data());
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                               tex_env_cubemap_buffer_, 0);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_buffer_);

        glDrawElements(draw_arrays_mode_, draw_arrays_size_, GL_UNSIGNED_INT,
                       nullptr);

        glDisableVertexAttribArray(0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    /** Final: Fill fine scale texture, then generate mipmap **/
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_env_cubemap_buffer_);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);

    glViewport(0, 0, view.GetWindowWidth(), view.GetWindowHeight());

    return true;
}

void HDRToEnvCubemapShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &triangle_buffer_);
        bound_ = false;
    }
}

bool HDRToEnvCubemapShader::PrepareRendering(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view) {

    /** Additional states **/
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    return true;
}

bool HDRToEnvCubemapShader::PrepareBinding(
    const geometry::Geometry &geometry,
    const RenderOption &option,
    const ViewControl &view,
    std::vector<Eigen::Vector3f> &points,
    std::vector<Eigen::Vector3i> &triangles) {

    /** Prepare camera **/
    projection_ = GLHelper::Perspective(90.0f, 1.0f, 0.1f, 10.0f);
    LoadViews(views_);

    /** Prepare geometry **/
    LoadCube(points, triangles);

    /** Prepare target texture **/
    tex_env_cubemap_buffer_ = CreateTextureCubemap(kCubemapSize, true, option);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(triangles.size() * 3);
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace open3d
