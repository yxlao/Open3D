//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "ShaderWrapperPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
class HDRToEnvCubemapShader : public ShaderWrapperPBR {
public:
    HDRToEnvCubemapShader() : HDRToEnvCubemapShader("HDRToCubemapShader") {}
    ~HDRToEnvCubemapShader() override { Release(); }

    GLuint GetGeneratedCubemapBuffer() const { return tex_env_cubemap_buffer_; }

protected:
    explicit HDRToEnvCubemapShader(const std::string &name)
        : ShaderWrapperPBR(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

    /** Dummy, load Cube instead **/
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    /** Dummy, texture for rendering is not used **/
    bool BindTextures(const std::vector<geometry::Image> &textures,
                      const RenderOption &option,
                      const ViewControl &view) final { return true; };
    /** Assign lighting **/
    bool BindLighting(const physics::Lighting &lighting,
                      const RenderOption &option,
                      const ViewControl &view) final;

    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;

    void UnbindGeometry() final;


protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view);
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points);

protected:
    /** locations **/
    /* array (cube) */
    GLuint vertex_position_;

    /* vertex shader */
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    GLuint tex_hdr_;

    /** buffers **/
    GLuint vertex_position_buffer_;

    const int kCubemapSize = 512;
    GLuint tex_env_cubemap_buffer_;    /* <- to be generated */

    /** cameras (fixed) **/
    GLHelper::GLMatrix4f projection_;
    std::vector<GLHelper::GLMatrix4f> views_;

    physics::IBLLighting ibl_;
};

}
}
}

