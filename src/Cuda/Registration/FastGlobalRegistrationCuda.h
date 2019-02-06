//
// Created by wei on 1/21/19.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Core/Geometry/PointCloud.h>
#include <Cuda/Geometry/PointCloudCuda.h>
#include <Cuda/Registration/FeatureExtractorCuda.h>
#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Geometry/NNCuda.h>

namespace open3d {
namespace cuda {

class FastGlobalRegistrationCudaDevice {
public:
    PointCloudCudaDevice source_;
    PointCloudCudaDevice target_;

    Array2DCudaDevice<float> source_features_;
    Array2DCudaDevice<float> target_features_;

    CorrespondenceSetCudaDevice corres_source_to_target_;
    CorrespondenceSetCudaDevice corres_target_to_source_;

    ArrayCudaDevice<Vector2i> corres_mutual_;
    ArrayCudaDevice<Vector2i> corres_final_;

    ArrayCudaDevice<float> results_;

    float par_;
    float scale_global_;

    __DEVICE__
    void ComputePointwiseJacobianAndResidual(
        int source_idx, int target_idx,
        Vector6f &jacobian_x, Vector6f &jacobian_y, Vector6f &jacobian_z,
        Vector3f &residual, float &lij);
};

class FastGlobalRegistrationCuda {
public:
    std::shared_ptr<FastGlobalRegistrationCudaDevice> device_ = nullptr;

public:
    FastGlobalRegistrationCuda() { Create(); }
    ~FastGlobalRegistrationCuda() { Release(); }
    void Create();
    void Release();

    void UpdateDevice();
    void ExtractResults(
        Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr, float &rmse);

public:
    void Initialize(PointCloud& source, PointCloud &target);
    double NormalizePointClouds();
    void AdvancedMatching();
    RegistrationResultCuda DoSingleIteration(int iter);
    RegistrationResultCuda ComputeRegistration();

public:
    PointCloudCuda source_;
    PointCloudCuda target_;

    FeatureExtractorCuda source_feature_extractor_;
    FeatureExtractorCuda target_feature_extractor_;

    Array2DCuda<float> source_features_;
    Array2DCuda<float> target_features_;

    NNCuda nn_source_to_target_;
    NNCuda nn_target_to_source_;

    CorrespondenceSetCuda corres_source_to_target_;
    CorrespondenceSetCuda corres_target_to_source_;

    ArrayCuda<Vector2i> corres_mutual_;
    ArrayCuda<Vector2i> corres_final_;

    ArrayCuda<float> results_;

    Eigen::Vector3d mean_source_;
    Eigen::Vector3d mean_target_;

    Eigen::Matrix4d transform_normalized_source_to_target_;
};

class FastGlobalRegistrationCudaKernelCaller {
public:
    static void ReciprocityTest(FastGlobalRegistrationCuda &fgr);
    static void TupleTest(FastGlobalRegistrationCuda &fgr);
    static void ComputeResultsAndTransformation(
        FastGlobalRegistrationCuda &fgr);
};

__GLOBAL__
void ReciprocityTestKernel(FastGlobalRegistrationCudaDevice server);
__GLOBAL__
void TupleTestKernel(FastGlobalRegistrationCudaDevice server,
                     ArrayCudaDevice<float> random_numbers,
                     int tuple_tests);
__GLOBAL__
void ComputeResultsAndTransformationKernel(
    FastGlobalRegistrationCudaDevice server);

} // cuda
} // open3d


