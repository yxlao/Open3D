//
// Created by wei on 4/13/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <stb_image/stb_image.h>

namespace open3d {
namespace geometry {

class Lighting : public Geometry2D {
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
    void Clear() override {};
    bool IsEmpty() const override {return false;};
    Eigen::Vector2d GetMinBound() const override {return Eigen::Vector2d(); };
    Eigen::Vector2d GetMaxBound() const override {return Eigen::Vector2d(); };

protected:
    explicit Lighting(LightingType type)
        : Geometry2D(Geometry::GeometryType::Lighting),
          type_(type) {}

private:
    LightingType type_;
};

class IBLLighting : public Lighting {
public:
    IBLLighting() : Lighting(LightingType::IBL) {}
    ~IBLLighting();

    bool ReadEnvFromHDR(const std::string &filename);

    Eigen::Vector3d SampleAround(const Eigen::Vector3d &position,
                                 double sigma);
    Eigen::Vector3f GetValueAt(const Eigen::Vector3d &position);
    void SetValueAt(const Eigen::Vector3d &position, const Eigen::Vector3f &value);
    void ShiftValue(const Eigen::Vector3d &direction,
                    const Eigen::Vector3f &psi);

public:
    std::shared_ptr<geometry::Image> hdr_; /** log - lat map **/
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

