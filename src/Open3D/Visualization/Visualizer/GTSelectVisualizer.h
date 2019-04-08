#pragma once
#include "Open3D/Visualization/Visualizer/VisualizerWithEditing.h"

namespace open3d {
namespace visualization {

class GTSelectVisualizer : public VisualizerWithEditing {
protected:
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
};

}  // namespace visualization
}  // namespace open3d
