//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include "InverseRendering/Visualization/Shader/ShaderWrapperPBR.h"
#include <InverseRendering/Geometry/TriangleMeshExtended.h>

namespace open3d {
namespace visualization {

namespace glsl {
/** Lighting should have been processed before being passed here **/
class DifferentialShader : public ShaderWrapperPBR {
public:
    DifferentialShader() : DifferentialShader("DifferentialShader") {}
    ~DifferentialShader() override { Release(); }

protected:
    explicit DifferentialShader(const std::string &name)
        : ShaderWrapperPBR(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;

    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool BindTextures(const std::vector<geometry::Image> &textures,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool BindLighting(const geometry::Lighting &lighting,
                      const RenderOption &option,
                      const ViewControl &view) final;

    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
public:
    void RebindTexture(const geometry::Image &image);
    void RebindGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        bool color, bool material, bool normal);
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
    GLuint tex_target_image_;
    GLuint camera_position_;
    GLuint viewport_;

    /** buffers **/
    GLuint vertex_position_buffer_;
    GLuint vertex_normal_buffer_;
    GLuint vertex_color_buffer_;
    GLuint vertex_material_buffer_;
    GLuint triangle_buffer_;

    /** Input **/
    std::vector<GLuint> tex_env_buffers_;

    /** Output **/
    const int kNumOutputTextures = 6;
    GLuint fbo_, rbo_;
    std::vector<GLuint> tex_fbo_buffers_;

public:
    bool is_debug_ = false;
    int target_img_id_ = 0;

    GLuint tex_target_img_buffer_;

    std::vector<std::shared_ptr<geometry::Image>> fbo_outputs_;
};

}
}
}
