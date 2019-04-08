#include "GTSelectVisualizer.h"

#include "Open3D/Visualization/Utility/PointCloudPicker.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Geometry/PointCloud.h"

namespace open3d {
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

}  // namespace visualization
}  // namespace open3d
