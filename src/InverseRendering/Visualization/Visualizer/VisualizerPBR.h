//
// Created by wei on 4/15/19.
//

#pragma once

#include <Open3D/Open3D.h>
#include <InverseRendering/Geometry/Lighting.h>

namespace open3d {
namespace visualization {

class VisualizerPBR : public VisualizerWithKeyCallback {
public:
    /** Dummy **/
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override {};

    /** In use **/
    virtual bool AddGeometryPBR(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        const std::vector<geometry::Image> &textures,
        const std::shared_ptr<geometry::Lighting> &lighting);

    virtual void Render() override;
};

class VisualizerDR : public VisualizerPBR {
public:
    virtual bool AddGeometryPBR(
        std::shared_ptr<geometry::Geometry> geometry_ptr,
        const std::vector<geometry::Image> &textures,
        const std::shared_ptr<geometry::Lighting> &lighting);
};
}
}


