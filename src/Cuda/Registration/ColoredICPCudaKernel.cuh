//
// Created by wei on 1/15/19.
//

#pragma once

#include "ColoredICPCuda.h"
#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <Cuda/Container/Array2DCudaDevice.cuh>

namespace open3d {
namespace cuda {

__global__
void ComputeColorGradientKernel(
    PointCloudCudaDevice pcl,
    CorrespondenceSetCudaDevice corres,
    ArrayCudaDevice<Vector3f> color_gradient) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= corres.indices_.size()) return;

    int i = corres.indices_[idx], j = 1;

    Vector3f &vt = pcl.points()[i];
    Vector3f &nt = pcl.normals()[i];
    Vector3f &color = pcl.colors()[i];
    float it = (color(0) + color(1) + color(2)) / 3.0f;

    Matrix3f AtA(0);
    Vector3f Atb(0);

    int nn = 0;
    while (corres.matrix_(i, j) != -1) {
        int adj_idx = corres.matrix_(i, j);
        Vector3f &vt_adj = pcl.points()[adj_idx];
        Vector3f &vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
        Vector3f &color_adj = pcl.colors()[adj_idx];
        float it_adj = (color_adj(0) + color_adj(1) + color_adj(2)) / 3.0f;

        float a0 = vt_proj(0) - vt(0);
        float a1 = vt_proj(1) - vt(1);
        float a2 = vt_proj(2) - vt(2);
        float b = it_adj - it;

        AtA(0, 0) += a0 * a0;
        AtA(0, 1) += a0 * a1;
        AtA(0, 2) += a0 * a2;
        AtA(1, 1) += a1 * a1;
        AtA(1, 2) += a1 * a2;
        AtA(2, 2) += a2 * a2;
        Atb(0) += a0 * b;
        Atb(1) += a1 * b;
        Atb(2) += a2 * b;

        ++nn;
    }

    /* orthogonal constraint */
    float nn2 = nn * nn;
    AtA(0, 0) += nn2 * nt(0) * nt(0);
    AtA(0, 1) += nn2 * nt(0) * nt(1);
    AtA(0, 2) += nn2 * nt(0) * nt(2);
    AtA(1, 1) += nn2 * nt(1) * nt(1);
    AtA(1, 2) += nn2 * nt(1) * nt(2);
    AtA(2, 2) += nn2 * nt(2) * nt(2);

    /* Symmetry */
    AtA(1, 0) = AtA(0, 1);
    AtA(2, 0) = AtA(0, 2);
    AtA(2, 1) = AtA(1, 2);

    color_gradient[i] = AtA.Solve(Atb);
}
} // cuda
} // open3d