//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "ShaderWrapperPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
class PreIntegrateLUTShader : public ShaderWrapperPBR {
public:
    PreIntegrateLUTShader() : PreIntegrateLUTShader("PreIntegrateLUTShader") {}
    ~PreIntegrateLUTShader() override { Release(); }

    GLuint GetGeneratedLUTBuffer() const { return tex_brdf_lut_buffer_; }

protected:
    explicit PreIntegrateLUTShader(const std::string &name)
        : ShaderWrapperPBR(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

    /** Dummy, load Quad instead **/
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    /** Dummy, texture for rendering is not used **/
    bool BindTextures(const std::vector<geometry::Image> &textures,
                      const RenderOption &option,
                      const ViewControl &view) final { return true; };
    /** Dummy **/
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
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector2f> &uvs);

protected:
    /** locations **/
    /* array (cube) */
    GLuint vertex_position_;
    GLuint vertex_uv_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_uv_buffer_;

    const int kTextureSize = 512;
    GLuint tex_brdf_lut_buffer_;    /* <- to be generated */
};

}
}
}

