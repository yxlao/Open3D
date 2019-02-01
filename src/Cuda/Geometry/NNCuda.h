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
    std::shared_ptr<NNCudaDevice> device_ = nullptr;

public:
    NNCuda();
    ~NNCuda();
    void UpdateDevice();

    /** Should be feature_size x feature_count
     *  This will encourage cache hits in parallel kernels
     **/
    void NNSearch(Eigen::MatrixXd &query,
                  Eigen::MatrixXd &reference);

    void NNSearch(Array2DCuda<float> &query,
                  Array2DCuda<float> &reference);

public:
    Array2DCuda<float> query_;
    Array2DCuda<float> reference_;

    Array2DCuda<int>   nn_idx_;
    Array2DCuda<float> nn_dist_;

    Array2DCuda<float> distance_matrix_;

    int query_blocks_;
    int ref_blocks_;
};

class NNCudaKernelCaller {
public:
    static void ComputeDistancesKernelCaller(NNCuda &nn);
    static void FindNNKernelCaller(NNCuda &nn);
};

__GLOBAL__
void ComputeDistancesKernel(NNCudaDevice nn);
__GLOBAL__
void FindNNKernel(NNCudaDevice nn);

} // cuda
} // open3d


