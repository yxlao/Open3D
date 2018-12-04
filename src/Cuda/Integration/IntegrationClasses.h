//
// Created by wei on 10/9/18.
//

#pragma once

#include <cstdlib>

namespace open3d {

namespace cuda {
template<size_t N>
class UniformTSDFVolumeCudaServer;

template<size_t N>
class UniformTSDFVolumeCuda;

template<size_t N>
class ScalableTSDFVolumeCudaServer;

template<size_t N>
class ScalableTSDFVolumeCuda;

template<size_t N>
class UniformMeshVolumeCudaServer;

template<size_t N>
class UniformMeshVolumeCuda;

template<size_t N>
class ScalableMeshVolumeCudaServer;

template<size_t N>
class ScalableMeshVolumeCuda;

} // cuda
} // open3d