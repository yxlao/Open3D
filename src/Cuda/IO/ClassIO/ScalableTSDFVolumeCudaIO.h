//
// Created by wei on 3/28/19.
//

#pragma once

#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>

namespace open3d {
namespace io {

template<size_t N>
bool WriteTSDFVolumeToBIN(const std::string &filename,
                          cuda::ScalableTSDFVolumeCuda<N> &volume);
template<size_t N>
bool ReadTSDFVolumeFromBIN(const std::string &filename,
                           cuda::ScalableTSDFVolumeCuda<N> &volume);

}
}


