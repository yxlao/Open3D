//
// Created by wei on 3/25/19.
//

#pragma once

#include <Open3D/Open3D.h>

namespace open3d {
namespace visualization {
class VisualizerWithCudaModule : public VisualizerWithKeyCallback {
public:
    /// Visualizer should be updated accordingly.
    virtual bool AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
};
} // visualization
} // open3d


