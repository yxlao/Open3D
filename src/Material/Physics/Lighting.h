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
    GLuint tex_cubemap_buffer_;
    GLuint tex_preconv_diffuse_buffer_;
    GLuint tex_prefilter_light_buffer_;
    GLuint tex_brdf_lut_buffer_;        /* (<H,V>, roughness) */

    void UpdateCubemapBuffer(GLuint tex_cubemap_buffer) {
        tex_cubemap_buffer_ = tex_cubemap_buffer;
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

