//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>

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

protected:
    explicit Lighting(LightingType type) : type_(type) {}

private:
    LightingType type_;
};

class IBLLighting : public Lighting {
public:
    IBLLighting() : Lighting(LightingType::IBL) {}
    ~IBLLighting() final = default;
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

