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
class SpotTextureMapShader : public ShaderWrapper {
public:
    SpotTextureMapShader() : SpotTextureMapShader("SpotTextureMapShader") {}
    ~SpotTextureMapShader() override { Release(); }

protected:

    explicit SpotTextureMapShader(const std::string &name)
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
    const int kNumTextures = 5;

    /** locations **/
    /* vertex shader */
    GLuint M_;
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    std::vector<GLuint> tex_symbols_;
    GLuint camera_position_;
    GLuint light_positions_;
    GLuint light_colors_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_normal_buffer_;
    GLuint vertex_uv_buffer_;
    GLuint triangle_buffer_;
    std::vector<GLuint> tex_buffers_;

    /** raw data **/
    std::vector<Eigen::Vector3f> light_positions_data_;
    std::vector<Eigen::Vector3f> light_colors_data_;
};

}
}
}


