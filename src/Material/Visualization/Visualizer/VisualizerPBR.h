//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <Material/Physics/Lighting.h>

namespace open3d {
namespace visualization {

class VisualizerPBR : public Visualizer {
public:
    /** Dummy **/
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override {};

    /** In use **/
    virtual bool AddGeometryPBR(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        const std::vector<geometry::Image> &textures,
        const std::shared_ptr<physics::Lighting> &lighting);

    virtual void Render() override;

protected:
    std::vector<std::vector<geometry::Image>> textures_;
    std::vector<std::shared_ptr<physics::Lighting>> lightings_;
};

}
}


