//
// Created by wei on 1/11/19.
//

#include "ColoredICPCuda.h"

namespace open3d {
namespace cuda {
void TransformationEstimationCudaForColoredICP::InitializeColorGradients(
    PointCloud &target,
    KDTreeFlann &kdtree,
    const KDTreeSearchParamHybrid &search_param) {

    /** Initialize correspondence matrix **/
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> corres_matrix
        = Eigen::Matrix<int, -1, -1, Eigen::RowMajor>::Constant(
            target.points_.size(), search_param.max_nn_, -1);

    /** OpenMP parallel K-NN search **/
#ifdef _OPENMP
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < target.points_.size(); ++i) {
            std::vector<int> indices(search_param.max_nn_);
            std::vector<double> dists(search_param.max_nn_);

            if (kdtree.SearchHybrid(target.points_[i],
                                    search_param.radius_, search_param.max_nn_,
                                    indices, dists) >= 3) {
                corres_matrix.block(i, 0, 1, indices.size()) =
                    Eigen::Map<Eigen::RowVectorXi>(
                        indices.data(), indices.size());
            }
        }
#ifdef _OPENMP
    }
#endif

    CorrespondenceSetCuda corres;
    corres.SetCorrespondenceMatrix(corres_matrix);
    corres.Compress();

    PointCloudCuda pcl;
    pcl.Upload(target);

    color_gradient_.Resize(target.points_.size());
    TransformationEstimationCudaForColoredICPKernelCaller::
    ComputeColorGradeintKernelCaller(pcl, corres, color_gradient_);
}
} // cuda
} // open3d
