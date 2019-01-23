//
// Created by wei on 1/23/19.
//

#pragma once

#include "FeatureCuda.h"

namespace open3d {
namespace cuda {
__device__
Vector4f FeatureCudaDevice::ComputePairFeature(int i, int j) {
    Vector4f result(0);

    const Vector3f &p1 = pcl_.points()[i];
    const Vector3f &p2 = pcl_.points()[j];
    Vector3f dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return result;
    }

    const Vector3f &n1 = pcl_.normals()[i];
    const Vector3f &n2 = pcl_.normals()[j];
    Vector3f n1_copy = n1;
    Vector3f n2_copy = n2;
    float angle1 = n1_copy.dot(dp2p1) / result(3);
    float angle2 = n2_copy.dot(dp2p1) / result(3);

    if (acosf(fabsf(angle1)) > acosf(fabsf(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }

    Vector3f v = dp2p1.cross(n1_copy);
    float v_norm = v.norm();
    if (v_norm == 0.0) {
        return Vector4f(0);
    }

    v /= v_norm;
    Vector3f w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2f(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

__device__
void FeatureCudaDevice::ComputeSPFHFeature(int i) {
    int nn = 0, max_nn = neighbors_.matrix_.max_cols_;
//
//    for (int j = 1; j < max_nn; ++j) {
//        int adj_idx = corres.matrix_(i, j);
//        if (adj_idx == -1) break;
//
//        Vector3f &vt_adj = pcl.points()[adj_idx];
//        Vector3f vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
//        Vector3f &color_adj = pcl.colors()[adj_idx];
//        float it_adj = (color_adj(0) + color_adj(1) + color_adj(2)) / 3.0f;
//
//        float a0 = vt_proj(0) - vt(0);
//        float a1 = vt_proj(1) - vt(1);
//        float a2 = vt_proj(2) - vt(2);
//        float b = it_adj - it;
//
//        AtA(0, 0) += a0 * a0;
//        AtA(0, 1) += a0 * a1;
//        AtA(0, 2) += a0 * a2;
//        AtA(1, 1) += a1 * a1;
//        AtA(1, 2) += a1 * a2;
//        AtA(2, 2) += a2 * a2;
//        Atb(0) += a0 * b;
//        Atb(1) += a1 * b;
//        Atb(2) += a2 * b;
//
//        ++nn;
}
} // cuda
} // open3d

