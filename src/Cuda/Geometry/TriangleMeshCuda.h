
//
// Created by wei on 10/10/18.
//

#pragma once
#include "VectorCuda.h"

namespace open3d {
class TriangleMeshCudaServer {
private:
    Vector3f *vertices_;
    Vector3f *vertex_normals_;
    Vector3f *vertex_colors_;

    Vector3i *triangles_;

public:
    int max_vertices_;
    int max_triangles_;

public:
    Vector3f &vertex(int i);
    Vector3f &normal(int i);
    Vector3f &color(int i);
    Vector3i &triangle(int i);
};

}