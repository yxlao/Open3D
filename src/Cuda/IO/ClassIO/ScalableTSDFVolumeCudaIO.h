//
// Created by wei on 3/28/19.
//

#pragma once

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

namespace open3d {
namespace io {

bool WriteTSDFVolumeToBIN(const std::string &filename,
                          cuda::ScalableTSDFVolumeCuda &volume);
bool ReadTSDFVolumeFromBIN(const std::string &filename,
                           cuda::ScalableTSDFVolumeCuda &volume,
                           int batch_size = 5000);

}
}


