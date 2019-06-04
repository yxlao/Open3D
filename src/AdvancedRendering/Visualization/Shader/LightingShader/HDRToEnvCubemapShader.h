//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Utility/BufferHelper.h"

namespace open3d {
namespace visualization {

namespace glsl {
class HDRToEnvCubemapShader : public ShaderWrapper {
public:
    HDRToEnvCubemapShader() : HDRToEnvCubemapShader("HDRToEnvCubemapShader") {}
    ~HDRToEnvCubemapShader() override { Release(); }

    GLuint GetGeneratedCubemapBuffer() const { return tex_env_cubemap_buffer_; }

protected:
    explicit HDRToEnvCubemapShader(const std::string &name)
        : ShaderWrapper(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

    /** Dummy, load Cube instead **/
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
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    /** locations **/
    /* vertex shader */
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    GLuint tex_hdr_symbol_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint triangle_buffer_;

    const int kCubemapSize = 512;
    GLuint tex_env_cubemap_buffer_;    /* <- to be generated */

    /** cameras (fixed) **/
    GLHelper::GLMatrix4f projection_;
    std::vector<GLHelper::GLMatrix4f> views_;
};

}
}
}

