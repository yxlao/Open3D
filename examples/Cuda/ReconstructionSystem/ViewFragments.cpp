//
// Created by wei on 2/5/19.
//

#include <IO/IO.h>
#include <Core/Core.h>
#include <Visualization/Visualization.h>
#include "System.h"

using namespace open3d;

int main(int argc, char **argv) {
    auto ply_filenames = GetFragmentPlyNames(kBasePath, kNumFragments);

    for (auto &ply_filename : ply_filenames) {
        auto pcl = CreatePointCloudFromFile(ply_filename);
        DrawGeometries({pcl});
    }
    return 0;
}