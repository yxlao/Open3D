//
// Created by wei on 1/21/19.
//

#pragma once

#include <Cuda/Common/UtilsCuda.h>
#include <Cuda/Container/Array2DCuda.h>
#include <Eigen/Eigen>

namespace open3d {
namespace cuda {

/**
 * This class is specially optimized for fpfh feature's 1-NN search
 */
class NNCudaDevice {
public:
    Array2DCudaDevice<float> query_; /* num_queries x feature_size */
    Array2DCudaDevice<float> ref_;   /* num_refs x feature_size */

    /* Local NN PER-BLOCK. Supports at most 256 blocks.
     * The final result should be reduced in the second pass. */
    Array2DCudaDevice<int>   nn_idx_;  /* num_queries x 256 */
    Array2DCudaDevice<float> nn_dist_; /* num_queries x 256 */

    /* We don't have enough shared memory -- it is the only way */
    Array2DCudaDevice<float> distance_matrix_;
};

class NNCuda {
public:
    std::shared_ptr<NNCudaDevice> server_ = nullptr;

public:
    NNCuda();
    ~NNCuda();
    void UpdateServer();

    void NNSearch(
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> &query,
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> &reference);

public:
    Array2DCuda<float> query_;
    Array2DCuda<float> reference_;

    Array2DCuda<int> nn_idx_;
    Array2DCuda<float> nn_dist_;

    Array2DCuda<float> distance_matrix_;

    int ref_blocks_;
};

class NNCudaKernelCaller {
public:
    static void ComputeAndReduceDistancesKernelCaller(NNCuda &nn);
    static void ReduceBlockwiseDistancesKernelCaller(NNCuda &nn);
};

__GLOBAL__
void ComputeAndReduceDistancesKernel(NNCudaDevice nn);

__GLOBAL__
void ReduceBlockwiseDistancesKernel(NNCudaDevice nn, int ref_blocks);

} // cuda
} // open3d


