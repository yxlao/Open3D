//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <stb_image/stb_image.h>

namespace open3d {
namespace physics {

class Lighting {
public:
    enum class LightingType {
        Spot = 0,
        Directional = 1,
        IBL = 2
    };

public:
    virtual ~Lighting() = default;
    LightingType GetLightingType() const {
        return type_;
    }

protected:
    explicit Lighting(LightingType type) : type_(type) {}

private:
    LightingType type_;
};

class IBLLighting : public Lighting {
public:
    IBLLighting() : Lighting(LightingType::IBL) {}
    ~IBLLighting();

    bool ReadDataFromHDR();

public:
    bool is_preprocessed_ = false;

    std::string filename_;

    GLuint tex_hdr_buffer_;
    GLuint tex_env_buffer_;
    GLuint tex_env_diffuse_buffer_;
    GLuint tex_env_specular_buffer_; /* roughness, R = 2<V, N>N - V */
    GLuint tex_lut_specular_buffer_; /* roughness, <V, N>) */

    void UpdateEnvBuffer(GLuint tex_env_buffer) {
        tex_env_buffer_ = tex_env_buffer;
    }
    void UpdateEnvDiffuseBuffer(GLuint tex_env_diffuse_buffer) {
        tex_env_diffuse_buffer_ = tex_env_diffuse_buffer;
    }
    void UpdateEnvSpecularBuffer(GLuint tex_env_specular_buffer) {
        tex_env_specular_buffer_ = tex_env_specular_buffer;
    }
    void UpdateLutSpecularBuffer(GLuint tex_lut_specular_buffer) {
        tex_lut_specular_buffer_ = tex_lut_specular_buffer;
    }
};

class SpotLighting : public Lighting {
public:
    SpotLighting() : Lighting(LightingType::Spot) {}
    ~SpotLighting() final = default;

public:
    std::vector<Eigen::Vector3f> light_positions_;
    std::vector<Eigen::Vector3f> light_colors_;
};

}
}

