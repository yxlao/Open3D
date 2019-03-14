//
// Created by wei on 2/5/19.
//

#include <Open3D/Open3D.h>
#include "examples/Cuda/DatasetConfig.h"

using namespace open3d;

int main(int argc, char **argv) {
    SetVerbosityLevel(utility::VerbosityLevel::VerboseDebug);

    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        kDefaultDatasetConfigDir + "/cmu/nsh.json";

    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (! is_success) return 1;

//    auto mesh = CreateMeshFromFile(config.GetReconstructedSceneFile());
//    mesh->ComputeTriangleNormals();
//    DrawGeometries({mesh});

    config.GetFragmentFiles();
    for (auto &ply_filename : config.fragment_files_) {
        auto pcl = io::CreatePointCloudFromFile(ply_filename);
        utility::PrintInfo("%s\n", ply_filename.c_str());
        visualization::DrawGeometries({pcl});
    }
    return 0;
}