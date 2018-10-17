//
// Created by wei on 9/27/18.
//

#ifndef OPEN3D_GEOMETRY_H
#define OPEN3D_GEOMETRY_H

#pragma once

#include <cstdlib>

namespace open3d {

template<typename T>
class ImageCudaServer;

template<typename T>
class ImageCuda;

template<typename T, size_t N>
class ImagePyramidCudaServer;

template<typename T, size_t N>
class ImagePyramidCuda;


enum VertexType {
    VertexRaw = 0,
    VertexWithNormal = 1,
    VertexWithColor = 2,
    VertexWithNormalAndColor = 3
};

template<VertexType type>
class TriangleMeshCudaServer;

template<VertexType type>
class TriangleMeshCuda;

class PointCloudCudaServer;

class PointCloudCuda;


}
#endif //OPEN3D_GEOMETRY_H
