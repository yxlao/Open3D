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
class IBLTextureMapShader : public ShaderWrapper {
public:
    IBLTextureMapShader() : IBLTextureMapShader("IBLTextureMapShader") {}
    ~IBLTextureMapShader() override { Release(); }

protected:
    explicit IBLTextureMapShader(const std::string &name)
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
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector2f> &uvs,
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    const int kNumObjectTextures = 5;
    const int kNumEnvTextures = 3;

    /* vertex shader */
    GLuint M_;
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    std::vector<GLuint> tex_object_symbols_; /* 5 textures for object */
    std::vector<GLuint> tex_env_symbols_;    /* 3 textures for env */
    GLuint camera_position_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_normal_buffer_;
    GLuint vertex_uv_buffer_;
    GLuint triangle_buffer_;

    std::vector<GLuint> tex_object_buffers_;
};

}
}
}


