//
// Created by wei on 2/4/19.
//

#include <iostream>
#include <Core/Core.h>
#include <IO/IO.h>
#include "DatasetConfig.h"

using namespace open3d;

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
        "/ReconstructionSystem/config/stonewall.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (! is_success) return 1;

    return 0;
}