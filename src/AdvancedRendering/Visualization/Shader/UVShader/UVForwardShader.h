//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "AdvancedRendering/Visualization/Utility/BufferHelper.h"
#include <AdvancedRendering/Geometry/ExtendedTriangleMesh.h>

namespace open3d {
namespace visualization {

namespace glsl {
/** Lighting should have been processed before being passed here **/
class UVForwardShader : public ShaderWrapper {
public:
    UVForwardShader() : UVForwardShader("UVForwardShader") {}
    ~UVForwardShader() override { Release(); }

protected:
    explicit UVForwardShader(const std::string &name)
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
                        std::vector<Eigen::Vector2f> &uvs,
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    const int kNumObjectTextures = 1;

    /* vertex shader */
    GLuint M_;
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    std::vector<GLuint> texes_object_symbols_; /* 1 texture for object */
    GLuint camera_position_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_uv_buffer_;
    GLuint triangle_buffer_;

    std::vector<GLuint> texes_object_buffers_;
};

}
}
}


