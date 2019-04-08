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

public:
    bool AddSubMeshesAndBeforeMesh(
            const std::vector<std::shared_ptr<const geometry::TriangleMesh>>
                    &sub_meshes,
            const std::shared_ptr<const geometry::TriangleMesh> &before_mesh);

protected:
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
    std::shared_ptr<geometry::PointCloud> merged_pcd_;
};

}  // namespace visualization
}  // namespace open3d
