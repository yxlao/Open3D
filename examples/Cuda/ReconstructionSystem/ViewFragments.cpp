//
// Created by wei on 2/5/19.
//

#include <IO/IO.h>
#include <Core/Core.h>
#include <Visualization/Visualization.h>
#include "DatasetConfig.h"

using namespace open3d;

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
        "/ReconstructionSystem/config/fr2_desktop.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (! is_success) return 1;

    auto mesh = CreateMeshFromFile(config.GetReconstructedSceneFile());
    DrawGeometries({mesh});

    config.GetFragmentFiles();
    for (auto &ply_filename : config.fragment_files_) {
        auto pcl = CreatePointCloudFromFile(ply_filename);
        DrawGeometries({pcl});
    }
    return 0;
}