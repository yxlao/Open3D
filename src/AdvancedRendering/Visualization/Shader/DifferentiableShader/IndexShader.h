//
// Created by wei on 4/24/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Utility/BufferHelper.h"
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>

namespace open3d {
namespace visualization {

namespace glsl {
/** Lighting should have been processed before being passed here **/
class IndexShader : public ShaderWrapper {
public:
    IndexShader() : IndexShader("IndexShader") {}
    ~IndexShader() override { Release(); }

protected:
    explicit IndexShader(const std::string &name)
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
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    /** locations **/
    /* vertex shader */
    GLuint V_;
    GLuint P_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint triangle_buffer_;

    GLuint tex_index_buffer_;

    GLuint fbo_;
    GLuint rbo_;

public:
    std::vector<std::shared_ptr<geometry::Image>> fbo_outputs_;
};

}
}
}

