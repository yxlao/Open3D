//
// Created by wei on 1/23/19.
//

#include "FeatureExtractorCuda.h"

namespace open3d {
namespace cuda {

FeatureExtractorCuda::FeatureExtractorCuda() {
    Create();
}

FeatureExtractorCuda::~FeatureExtractorCuda() {
    Release();
}

FeatureExtractorCuda::FeatureExtractorCuda(const FeatureExtractorCuda &other) {
    device_ = other.device_;
    neighbors_ = other.neighbors_;
    spfh_features_ = other.spfh_features_;
    fpfh_features_ = other.fpfh_features_;
    pcl_ = other.pcl_;
}

FeatureExtractorCuda& FeatureExtractorCuda::operator=(const FeatureExtractorCuda &other) {
    if (this != &other) {
        device_ = other.device_;
        neighbors_ = other.neighbors_;
        spfh_features_ = other.spfh_features_;
        fpfh_features_ = other.fpfh_features_;
        pcl_ = other.pcl_;
    }
    return *this;
}

void FeatureExtractorCuda::Create() {
    if (device_ == nullptr) {
        device_ = std::make_shared<FeatureCudaDevice>();
    }
}

void FeatureExtractorCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        pcl_.Release();
        neighbors_.Release();
        spfh_features_.Release();
        fpfh_features_.Release();
    }

    device_ = nullptr;
}

void FeatureExtractorCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->pcl_ = *pcl_.device_;
        device_->neighbors_ = *neighbors_.device_;
        device_->spfh_features_ = *spfh_features_.device_;
        device_->fpfh_features_ = *fpfh_features_.device_;
    }
}

void FeatureExtractorCuda::Compute(
    PointCloud &pcl, const KDTreeSearchParamHybrid &param) {
    pcl_.Create(VertexWithNormal, pcl.points_.size());
    pcl_.Upload(pcl);

    spfh_features_.Create(33, pcl.points_.size());
    spfh_features_.Memset(0);
    fpfh_features_.Create(33, pcl.points_.size());
    fpfh_features_.Memset(0);

    KDTreeFlann kdtree;
    kdtree.SetGeometry(pcl);

    /** Initialize correspondence matrix for neighbors **/
    Eigen::MatrixXi corres_matrix = Eigen::MatrixXi::Constant(
        param.max_nn_, pcl.points_.size(), -1);

    /** OpenMP parallel K-NN search **/
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < (int)pcl.points_.size(); ++i) {
            std::vector<int> indices(param.max_nn_);
            std::vector<double> dists(param.max_nn_);

            if (kdtree.SearchHybrid(pcl.points_[i],
                param.radius_, param.max_nn_, indices, dists) >= 3) {
                corres_matrix.block(0, i, indices.size(), 1) =
                    Eigen::Map<Eigen::VectorXi>(
                        indices.data(), indices.size());
            }
        }
#ifdef _OPENMP
    }
#endif

    neighbors_.SetCorrespondenceMatrix(corres_matrix);
    neighbors_.Compress();
    UpdateDevice();

    FeatureCudaKernelCaller::ComputeSPFHFeature(*this);
    FeatureCudaKernelCaller::ComputeFPFHFeature(*this);
}
} // cuda
} // open3d