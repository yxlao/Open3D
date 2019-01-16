//
// Created by wei on 1/11/19.
//

#include "ColoredICPCuda.h"
#include <Core/Core.h>

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

    /** CPU matrix to CUDA Array2D **/
    Timer timer;
    timer.Start();
    CorrespondenceSetCuda corres;
    corres.SetCorrespondenceMatrix(corres_matrix);
    timer.Stop();
    PrintInfo("Setting correspondences takes %f ms\n", timer.GetDuration());

    timer.Start();
    corres.Compress();
    timer.Stop();
    PrintInfo("Compression takes %f ms\n", timer.GetDuration());

    PointCloudCuda pcl;
    timer.Start();
    pcl.Create(VertexWithNormalAndColor, target.points_.size());
    pcl.Upload(target);
    timer.Stop();
    PrintInfo("Upload takes %f ms\n", timer.GetDuration());

    timer.Start();
    color_gradient_.Resize(target.points_.size());
    TransformationEstimationCudaForColoredICPKernelCaller::
    ComputeColorGradeintKernelCaller(pcl, corres, color_gradient_);
    timer.Stop();
    PrintInfo("Compute gradient takes %f ms\n", timer.GetDuration());
}
} // cuda
} // open3d
