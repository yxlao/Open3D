//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <Material/Physics/Lighting.h>

namespace open3d {
namespace visualization {

namespace glsl {

class ShaderWrapperPBR : public ShaderWrapper {
public:
    ~ShaderWrapperPBR() override = default;

protected:
    explicit ShaderWrapperPBR(const std::string &name) : ShaderWrapper(name) {};

    /** Wrappers for more organized data preparation **/
    template<typename T>
    GLuint BindBuffer(const std::vector<T> &vec,
                      const GLuint &buffer_type,
                      const visualization::RenderOption &option) {
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(buffer_type, buffer);
        glBufferData(buffer_type, vec.size() * sizeof(T),
                     vec.data(), GL_STATIC_DRAW);
        return buffer;
    }
    GLuint BindTexture2D(const geometry::Image &texture,
                         const visualization::RenderOption &option);
    GLuint CreateTexture2D(GLuint width, GLuint height, bool use_mipmap,
                           const visualization::RenderOption &option);

    GLuint BindTextureCubemap(const std::vector<geometry::Image> &textures,
                              const visualization::RenderOption &option);
    GLuint CreateTextureCubemap(GLuint size, bool use_mipmap,
                                const visualization::RenderOption &option);

public:
    void LoadCube(std::vector<Eigen::Vector3f> &vertices,
                  std::vector<Eigen::Vector3i> &triangles);
    void LoadQuad(std::vector<Eigen::Vector3f> &vertices,
                  std::vector<Eigen::Vector2f> &uvs);
    void LoadViews(std::vector<GLHelper::GLMatrix4f> &views);

public:
    bool Render(const geometry::Geometry &geometry,
                const std::vector<geometry::Image> &textures,
                const physics::Lighting &lighting,
                const RenderOption &option,
                const ViewControl &view);

    /** Bind is separated in three functions **/
    /* virtual bool BindGeometry */
    virtual bool BindTextures(const std::vector<geometry::Image> &textures,
                              const RenderOption &option,
                              const ViewControl &view) = 0;
    virtual bool BindLighting(const physics::Lighting &lighting,
                              const RenderOption &option,
                              const ViewControl &view) = 0;

    /** Render requires only geometry: others are prestored **/
    /* virtual bool RenderGeometry */

    /** Unbind requires only geometry: others are unbind together **/
    /* virtual bool UnbindGeometry */
};
}
}
}


