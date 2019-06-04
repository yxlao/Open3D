//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Utility/BufferHelper.h"

namespace open3d {
namespace visualization {

namespace glsl {
class SimpleTextureShader : public ShaderWrapper {
public:
    SimpleTextureShader() : SimpleTextureShader("SimpleTextureShader") {}
    ~SimpleTextureShader() override { Release(); }

protected:
    explicit SimpleTextureShader(const std::string &name)
        : ShaderWrapper(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

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
    GLuint texture_symbol_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_uv_buffer_;
};

}
}
}

