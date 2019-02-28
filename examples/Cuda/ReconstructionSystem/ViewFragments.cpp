//
// Created by wei on 2/5/19.
//

#include <IO/IO.h>
#include <Core/Core.h>
#include <Visualization/Visualization.h>
#include "examples/Cuda/DatasetConfig.h"

using namespace open3d;

int main(int argc, char **argv) {
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
        "/ReconstructionSystem/config/office3.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (! is_success) return 1;

    auto mesh = CreateMeshFromFile(config.GetReconstructedSceneFile());
    mesh->ComputeTriangleNormals();
    DrawGeometries({mesh});

    config.GetFragmentFiles();
    for (int i = 0; i < config.fragment_files_.size(); ++i) {
        auto ply_filename = config.fragment_files_[i];
        auto pcl = CreatePointCloudFromFile(ply_filename);
        DrawGeometries({pcl}, ply_filename);
    }
    return 0;
}