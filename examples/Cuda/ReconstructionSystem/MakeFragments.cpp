//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
using namespace open3d;

std::shared_ptr<PointCloud> ProcessSingleFragment(
    int fragment_id,
    const std::vector<std::pair<std::string, std::string>> &filenames) {
    // odometry
    // posegraph optimization (unnecessary, but can be added here)
    // integration per fragment comes here
}

