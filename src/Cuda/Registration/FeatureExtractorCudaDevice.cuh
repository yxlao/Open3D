//
// Created by wei on 1/23/19.
//

#pragma once

#include "FeatureExtractorCuda.h"

namespace open3d {
namespace cuda {
__device__
Vector4f FeatureCudaDevice::ComputePairFeature(int i, int j) {
    Vector4f result(0);

    const Vector3f &p1 = pcl_.points_[i];
    const Vector3f &p2 = pcl_.points_[j];
    Vector3f dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return result;
    }

    const Vector3f &n1 = pcl_.normals_[i];
    const Vector3f &n2 = pcl_.normals_[j];
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
void FeatureCudaDevice::ComputeSPFHFeature(int i, int max_nn) {
    for (int k = 1; k < max_nn; ++k) {
        int j = neighbors_.matrix_(k, i);

        float hist_incr = 100.0f / (max_nn - 1);
        Vector4f pf = ComputePairFeature(i, j);
        int h_index = (int)(floorf(11.0f * (pf(0) + M_PIf) / (2.0f * M_PIf)));
        if (h_index < 0) h_index = 0;
        if (h_index >= 11) h_index = 10;
        spfh_features_(h_index, i) += hist_incr;
        h_index = (int)(floorf(11.0f * (pf(1) + 1.0f) * 0.5f));
        if (h_index < 0) h_index = 0;
        if (h_index >= 11) h_index = 10;
        spfh_features_(h_index + 11, i) += hist_incr;
        h_index = (int)(floorf(11.0f * (pf(2) + 1.0f) * 0.5f));
        if (h_index < 0) h_index = 0;
        if (h_index >= 11) h_index = 10;
        spfh_features_(h_index + 22, i) += hist_incr;
    }
}

__device__
void FeatureCudaDevice::ComputeFPFHFeature(int i, int max_nn) {
    float sum[3] = {0, 0, 0};
    Vector3f &pi = pcl_.points_[i];

    /** Add up neighbor's spfh **/
    for (int k = 1; k < max_nn; ++k) {
        int j = neighbors_.matrix_(k, i);

        Vector3f &pj = pcl_.points_[j];
        Vector3f pij = pi - pj;
        float dist = pij.dot(pij);

        if (dist == 0) continue;
        for (int f = 0; f < 33; ++f) {
            float val = spfh_features_(f, j) / dist;
            sum[f / 11] += val;
            fpfh_features_(f, i) += val;
        }
    }

    sum[0] = sum[0] != 0 ? 100.0f / sum[0] : 0;
    sum[1] = sum[1] != 0 ? 100.0f / sum[1] : 0;
    sum[2] = sum[2] != 0 ? 100.0f / sum[2] : 0;

    for (int f = 0; f < 33; ++f) {
        fpfh_features_(f, i) *= sum[f / 11];
        fpfh_features_(f, i) += spfh_features_(f, i);
    }
}
} // cuda
} // open3d

