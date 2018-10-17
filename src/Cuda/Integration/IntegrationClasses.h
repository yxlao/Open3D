//
// Created by wei on 10/9/18.
//

#pragma once

#include <cstdlib>

namespace open3d {
template<size_t N>
class UniformTSDFVolumeCudaServer;

template<size_t N>
class UniformTSDFVolumeCuda;

enum VertexType {
    VertexRaw = 0,
    VertexWithColor = 1,
    VertexWithNormal = 2,
    VertexWithNormalAndColor = 3
};

template<VertexType type, size_t N>
class UniformMeshVolumeCudaServer;

template<VertexType type, size_t N>
class UniformMeshVolumeCuda;
}