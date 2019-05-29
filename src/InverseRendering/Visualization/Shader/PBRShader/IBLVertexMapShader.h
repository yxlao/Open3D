//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "InverseRendering/Visualization/Shader/ShaderWrapperPBR.h"
#include <InverseRendering/Geometry/ExtendedTriangleMesh.h>

namespace open3d {
namespace visualization {

namespace glsl {
/** Lighting should have been processed before being passed here **/
class IBLVertexMapShader : public ShaderWrapperPBR {
public:
    IBLVertexMapShader() : IBLVertexMapShader("IBLNoTexShader") {}
    ~IBLVertexMapShader() override { Release(); }

protected:
    explicit IBLVertexMapShader(const std::string &name)
        : ShaderWrapperPBR(name) { Compile(); }

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

public:
    void RebindGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        bool color, bool material, bool normal);

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view);
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        std::vector<Eigen::Vector3f> &points,
                        std::vector<Eigen::Vector3f> &normals,
                        std::vector<Eigen::Vector3f> &colors,
                        std::vector<Eigen::Vector3f> &materials,
                        std::vector<Eigen::Vector3i> &triangles);

protected:
    /** locations **/
    /* vertex shader */
    GLuint M_;
    GLuint V_;
    GLuint P_;

    /* fragment shader */
    const int kNumEnvTextures = 3;
    std::vector<GLuint> texes_env_;    /* 3 textures for env */
    GLuint camera_position_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_normal_buffer_;
    GLuint vertex_color_buffer_;
    GLuint vertex_material_buffer_;
    GLuint triangle_buffer_;

    /** Input **/
    std::vector<GLuint> tex_env_buffers_;
};

}
}
}


