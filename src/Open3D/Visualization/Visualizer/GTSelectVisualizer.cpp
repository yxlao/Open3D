#include "GTSelectVisualizer.h"

#include "Open3D/Visualization/Utility/PointCloudPicker.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"

#include <numeric>
#include <vector>

namespace open3d {

geometry::PointCloud mesh_to_pcd(const geometry::TriangleMesh &mesh) {
    geometry::PointCloud pcd;
    pcd.points_ = mesh.vertices_;
    pcd.colors_ = mesh.vertex_colors_;
    pcd.normals_ = mesh.vertex_normals_;
    return pcd;
}

namespace visualization {

void GTSelectVisualizer::MouseButtonCallback(GLFWwindow *window,
                                             int button,
                                             int action,
                                             int mods) {
    auto &view_control = (ViewControlWithEditing &)(*view_control_ptr_);
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE &&
        (mods & GLFW_MOD_SHIFT)) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
#ifdef __APPLE__
        x /= pixel_to_screen_coordinate_;
        y /= pixel_to_screen_coordinate_;
#endif
        int index = PickPoint(x, y);
        if (index == -1) {
            utility::PrintInfo("No point has been picked.\n");
        } else {
            const auto &point =
                    ((const geometry::PointCloud &)(*geometry_ptrs_[0]))
                            .points_[index];
            utility::PrintInfo(
                    "Picked point #%d (%.2f, %.2f, %.2f) to add in "
                    "queue.\n",
                    index, point(0), point(1), point(2));
            pointcloud_picker_ptr_->picked_indices_.push_back((size_t)index);
            is_redraw_required_ = true;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE &&
               (mods & GLFW_MOD_SHIFT)) {
        if (pointcloud_picker_ptr_->picked_indices_.empty() == false) {
            utility::PrintInfo("Remove picked point #%d from pick queue.\n",
                               pointcloud_picker_ptr_->picked_indices_.back());
            pointcloud_picker_ptr_->picked_indices_.pop_back();
            is_redraw_required_ = true;
        }
    }
    Visualizer::MouseButtonCallback(window, button, action, mods);
}

bool GTSelectVisualizer::UpdateMergedPointCloud(
        const std::vector<size_t> &sub_mesh_select_index) {
    bool rc = true;
    if (merged_pcd_ == nullptr) {
        merged_pcd_ = std::make_shared<geometry::PointCloud>();
        rc = AddGeometry(merged_pcd_);
    }
    if (!rc) {
        return rc;
    }

    // merged_pcd_->Clear();
    // for (const size_t &select_index : sub_mesh_select_index) {
    //     auto sub_mesh = sub_meshes_[select_index];
    //     merged_pcd_->operator+=(mesh_to_pcd(*sub_mesh));
    // }
    // merged_pcd_->operator+=(mesh_to_pcd(*before_mesh_));
    // utility::PrintInfo("merged_pcd_.points_.size() %d\n",
    //                    merged_pcd_->points_.size());

    // original_geometry_ptr_ = merged_pcd_;
    // auto ptr = std::make_shared<geometry::PointCloud>();
    // *ptr = (const geometry::PointCloud &)*original_geometry_ptr_;
    // editing_geometry_ptr_ = ptr;
    // editing_geometry_renderer_ptr_ =
    //         std::make_shared<glsl::PointCloudRenderer>();
    // if (editing_geometry_renderer_ptr_->AddGeometry(editing_geometry_ptr_) ==
    //     false) {
    //     return false;
    // }
    // ResetViewPoint(true);
    // (geometry::PointCloud &)*editing_geometry_ptr_ =
    //         (const geometry::PointCloud &)*original_geometry_ptr_;
    // editing_geometry_renderer_ptr_->UpdateGeometry();
    // is_redraw_required_ = true;
    // UpdateWindowTitle();
    UpdateGeometry();
    return rc;
}

bool GTSelectVisualizer::AddSubMeshesAndBeforeMesh(
        const std::vector<std::shared_ptr<const geometry::TriangleMesh>>
                &sub_meshes,
        const std::shared_ptr<const geometry::TriangleMesh> &before_mesh) {
    sub_meshes_ = sub_meshes;
    before_mesh_ = before_mesh;
    // std::vector<size_t> sub_mesh_select_index(
    //         sub_meshes_.size());  // vector with 100 ints.
    // std::iota(std::begin(sub_mesh_select_index),
    //           std::end(sub_mesh_select_index), 0);

    // Add merged_pcd_
    merged_pcd_ = std::make_shared<geometry::PointCloud>();
    for (const std::shared_ptr<const geometry::TriangleMesh> &sub_mesh :
         sub_meshes_) {
        merged_pcd_->operator+=(mesh_to_pcd(*sub_mesh));
    }
    merged_pcd_->operator+=(mesh_to_pcd(*before_mesh_));
    return AddGeometry(merged_pcd_);
}

}  // namespace visualization
}  // namespace open3d
