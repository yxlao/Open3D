#pragma once
#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"

namespace open3d {

namespace geometry {
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace visualization {

class GTSelectVisualizer : public VisualizerWithEditing {
public:
    GTSelectVisualizer(double voxel_size = -1.0,
                       bool use_dialog = true,
                       const std::string &directory = "")
        : VisualizerWithEditing(voxel_size, use_dialog, directory) {}

    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;

protected:
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
};

}  // namespace visualization
}  // namespace open3d
