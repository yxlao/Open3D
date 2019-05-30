//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Shader/ShaderWrapperPBR.h"

namespace open3d {
namespace visualization {

namespace glsl {
class PreIntegrateLUTSpecularShader : public ShaderWrapperPBR {
public:
    PreIntegrateLUTSpecularShader() : PreIntegrateLUTSpecularShader("PreIntegrateLUTShader") {}
    ~PreIntegrateLUTSpecularShader() override { Release(); }

    GLuint GetGeneratedLUTBuffer() const { return tex_lut_specular_buffer_; }

protected:
    explicit PreIntegrateLUTSpecularShader(const std::string &name)
        : ShaderWrapperPBR(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

    /** Dummy, load Quad instead **/
    bool BindGeometry(const geometry::Geometry &geometry,
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
    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_uv_buffer_;

    const int kTextureSize = 512;
    GLuint tex_lut_specular_buffer_;    /* <- to be generated */
};

}
}
}

