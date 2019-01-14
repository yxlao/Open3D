//
// Created by wei on 1/11/19.
//

#pragma once

#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/TransformEstimationCuda.h>
#include <Core/Geometry/KDTreeFlann.h>

namespace open3d {
namespace cuda {

class TransformationEstimationCudaForColoredICPDevice {

};

class TransformationEstimationCudaForColoredICP
    : public TransformationEstimationCuda {

public:
    TransformationEstimationType GetTransformationEstimationType()
    const override { return type_; };

    TransformationEstimationCudaForColoredICP(
        float lambda_geometric = 0.968f) :
        lambda_geometric_(lambda_geometric) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0)
            lambda_geometric_ = 0.968f;
    }
    ~TransformationEstimationCudaForColoredICP() override {}

public:
    /** 1. Evaluate fitness and rmse of the previous transformation,
     *  2. Build linear system to solve for updated transformation **/
    RegistrationResultCuda ComputeResultsAndTransformation(
        const PointCloudCuda &source,
        const PointCloudCuda &target,
        const CorrespondenceSetCuda &corres) const override;

    /** Computes color gradients
     * 1. Get correspondence matrix on CPU
     * 2. Compress the correspondence matrix
     * 3. Use the compressed correspondence matrix to build linear systems
     * and compute color gradients.
     * **/
    void InitializePointCloudForColoredICP(
        PointCloud &target,
        KDTreeFlann &kdtree,
        KDTreeSearchParamHybrid &search_param);

public:
    float lambda_geometric_;
    ArrayCuda<Vector3f> color_gradient_;
    ArrayCuda<float> results_;

private:
    const TransformationEstimationType type_ =
        TransformationEstimationType::ColoredICP;
};

}
}


