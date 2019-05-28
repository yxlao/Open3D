//
// Created by wei on 4/15/19.
//

#pragma once

#include <InverseRendering/Geometry/Lighting.h>
#include <Open3D/Open3D.h>

namespace open3d {
namespace visualization {

class VisualizerPBR : public VisualizerWithKeyCallback {
public:
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override {};
//    virtual void BuildLighting(
//        std::shared_ptr<geometry::Lighting> &lighting);

    /** In use **/
    virtual bool AddGeometryPBR(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        const std::shared_ptr<geometry::Lighting> &lighting);

    virtual void Render() override;
};

class VisualizerDR : public VisualizerPBR {
public:
    virtual bool AddGeometryPBR(
            std::shared_ptr<geometry::Geometry> geometry_ptr,
            const std::shared_ptr<geometry::Lighting> &lighting);

    bool CaptureBuffer(const std::string &filename, int index);

    bool SetTargetImage(const geometry::Image &target,
                        const camera::PinholeCameraParameters &view);
    bool UpdateLighting();
    float CallSGD(float lambda,
                  bool update_albedo,
                  bool update_material,
                  bool update_normal);
};
}  // namespace visualization
}  // namespace open3d
