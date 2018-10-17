//
// Created by wei on 10/9/18.
//

#pragma once

#include <Cuda/Geometry/GeometryClasses.h>
#include <cstdlib>

namespace open3d {
template<size_t N>
class UniformTSDFVolumeCudaServer;

template<size_t N>
class UniformTSDFVolumeCuda;

template<VertexType type, size_t N>
class UniformMeshVolumeCudaServer;

template<VertexType type, size_t N>
class UniformMeshVolumeCuda;
}