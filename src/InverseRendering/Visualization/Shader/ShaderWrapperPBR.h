//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/Lighting.h>

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
                      const GLuint &buffer_type ,
                      const RenderOption &option) {
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(buffer_type, buffer);
        glBufferData(buffer_type, vec.size() * sizeof(T),
                     vec.data(), GL_STATIC_DRAW);
        return buffer;
    }

    GLuint BindTexture2D(const geometry::Image &texture,
                         const visualization::RenderOption &option);
    bool BindTexture2D(GLuint &texture_id,
                       const geometry::Image &texture,
                       const visualization::RenderOption &option);

    GLuint CreateTexture2D(
        GLuint width, GLuint height,
        GLenum internal_format, GLenum format, GLenum type,
        bool use_mipmap, const visualization::RenderOption &option);

    /** A little bit redundant **/
    std::shared_ptr<geometry::Image> ReadTexture2D(
        GLuint width, GLuint height, int channels, int bytes_per_channel,
        GLenum format, GLenum type);

    GLuint BindTextureCubemap(const std::vector<geometry::Image> &textures,
                              const visualization::RenderOption &option);
    GLuint CreateTextureCubemap(GLuint size, bool use_mipmap,
                                const visualization::RenderOption &option);

public:
    bool Render(const geometry::Geometry &geometry,
                const std::vector<geometry::Image> &textures,
                const geometry::Lighting &lighting,
                const RenderOption &option,
                const ViewControl &view);

    /** Bind is separated in three functions **/
    /* virtual bool BindGeometry */
    virtual bool BindTextures(const std::vector<geometry::Image> &textures,
                              const RenderOption &option,
                              const ViewControl &view) = 0;
    virtual bool BindLighting(const geometry::Lighting &lighting,
                              const RenderOption &option,
                              const ViewControl &view) = 0;

    /** Render requires only geometry: others are prestored **/
    /* virtual bool RenderGeometry */

    /** Unbind requires only geometry: others are unbind together **/
    /* virtual bool UnbindGeometry */

    /** Utility: primitives **/
public:
    void LoadCube(std::vector<Eigen::Vector3f> &vertices,
                  std::vector<Eigen::Vector3i> &triangles);
    void LoadQuad(std::vector<Eigen::Vector3f> &vertices,
                  std::vector<Eigen::Vector2f> &uvs);
    void LoadViews(std::vector<GLHelper::GLMatrix4f> &views);

public:
    bool CheckGLState(const std::string &msg) {
        GLenum ret = glGetError();
        if (ret != GL_NO_ERROR) {
            utility::PrintWarning(
                "[OpenGL error]: %d at %s\n", ret, msg.c_str());
        }
    }
};
}
}
}


