//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Utility/BindWrapper.h"

namespace open3d {
namespace visualization {

namespace glsl {
class BackgroundShader : public ShaderWrapper {
public:
    BackgroundShader() : BackgroundShader("BackgroundShader") {}
    ~BackgroundShader() override { Release(); }

protected:

    explicit BackgroundShader(const std::string &name)
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
                          const ViewControl &view) { return true; }
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    /** locations **/
    GLuint V_;               /* vertex shader */
    GLuint P_;
    GLuint tex_env_;         /* fragment shader */

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint triangle_buffer_;

    GLuint tex_env_buffer_;
};

}
}
}

