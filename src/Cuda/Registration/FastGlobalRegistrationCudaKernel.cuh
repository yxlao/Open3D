//
// Created by wei on 1/21/19.
//

#pragma once

#include "FastGlobalRegistrationCuda.h"
#include "FastGlobalRegistrationCudaDevice.cuh"
#include <Cuda/Container/ArrayCudaDevice.cuh>
#include <cuda.h>
#include <curand.h>

namespace open3d {
namespace cuda {

__global__
void ReciprocityTestKernel(FastGlobalRegistrationCudaDevice server) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.corres_source_to_target_.indices_.size()) return;

    int i = server.corres_source_to_target_.indices_[idx];
    int j = server.corres_source_to_target_.matrix_(0, i);
    if (j != -1 && server.corres_target_to_source_.matrix_(0, j) == i) {
        server.corres_mutual_.push_back(Vector2i(i, j));
    }
}

void FastGlobalRegistrationCudaKernelCaller::ReciprocityTest(
    FastGlobalRegistrationCuda &fgr) {
    fgr.corres_mutual_.set_iterator(0);

    const dim3 blocks(
        DIV_CEILING(fgr.corres_source_to_target_.indices_.size(),
            THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ReciprocityTestKernel<<<blocks, threads>>>(*fgr.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void TupleTestKernel(FastGlobalRegistrationCudaDevice server,
                     ArrayCudaDevice<float> random_numbers,
                     int tuple_tests) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= tuple_tests) return;

    const int corres_count = server.corres_mutual_.size();
    const int rand0 = (int) (corres_count * random_numbers[idx * 3 + 0]);
    const int rand1 = (int) (corres_count * random_numbers[idx * 3 + 1]);
    const int rand2 = (int) (corres_count * random_numbers[idx * 3 + 2]);

    Vector2i &pair0 = server.corres_mutual_[rand0];
    Vector2i &pair1 = server.corres_mutual_[rand1];
    Vector2i &pair2 = server.corres_mutual_[rand2];

    // collect 3 points from i-th fragment
    Vector3f &pti0 = server.source_.points_[pair0(0)];
    Vector3f &pti1 = server.source_.points_[pair1(0)];
    Vector3f &pti2 = server.source_.points_[pair2(0)];
    float li0 = (pti0 - pti1).norm();
    float li1 = (pti1 - pti2).norm();
    float li2 = (pti2 - pti0).norm();

    // collect 3 points from j-th fragment
    Vector3f &ptj0 = server.target_.points_[pair0(1)];
    Vector3f &ptj1 = server.target_.points_[pair1(1)];
    Vector3f &ptj2 = server.target_.points_[pair2(1)];
    float lj0 = (ptj0 - ptj1).norm();
    float lj1 = (ptj1 - ptj2).norm();
    float lj2 = (ptj2 - ptj0).norm();

    const float scale = 0.95f;
    if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
        (li1 * scale < lj1) && (lj1 < li1 / scale) &&
        (li2 * scale < lj2) && (lj2 < li2 / scale)) {
        server.corres_final_.push_back(pair0);
        server.corres_final_.push_back(pair1);
        server.corres_final_.push_back(pair2);
    }
}

void FastGlobalRegistrationCudaKernelCaller::TupleTest(
    FastGlobalRegistrationCuda &fgr) {

    size_t tuple_tests = fgr.corres_mutual_.size() * 100;
    size_t n = tuple_tests * 3;

    ArrayCuda<float> random_numbers;
    random_numbers.Create(n);

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, (unsigned int) std::time(0));
    curandGenerateUniform(gen, random_numbers.device_->data(), n);
    curandDestroyGenerator(gen);

    const dim3 blocks(DIV_CEILING(tuple_tests, THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);

    TupleTestKernel << < blocks, threads >> > (
        *fgr.device_, *random_numbers.device_, tuple_tests);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeMeanKernel(PointCloudCudaDevice server,
                       ArrayCudaDevice<Vector3f> mean) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.points_.size()) return;

    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    Vector3f &vertex = server.points_[idx];
    local_sum0[tid] = vertex(0);
    local_sum1[tid] = vertex(1);
    local_sum2[tid] = vertex(2);
    __syncthreads();

    if (tid < 128) {
        local_sum0[tid] += local_sum0[tid + 128];
        local_sum1[tid] += local_sum1[tid + 128];
        local_sum2[tid] += local_sum2[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        local_sum0[tid] += local_sum0[tid + 64];
        local_sum1[tid] += local_sum1[tid + 64];
        local_sum2[tid] += local_sum2[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        WarpReduceSum<float>(local_sum0, tid);
        WarpReduceSum<float>(local_sum1, tid);
        WarpReduceSum<float>(local_sum2, tid);
    }

    if (tid == 0) {
        atomicAdd(&mean[0](0), local_sum0[0]);
        atomicAdd(&mean[0](1), local_sum1[0]);
        atomicAdd(&mean[0](2), local_sum2[0]);
    }
    __syncthreads();
}

Eigen::Vector3d FastGlobalRegistrationCudaKernelCaller::ComputePointCloudSum(
    PointCloudCuda &pcl) {

    ArrayCuda<Vector3f> mean;
    mean.Create(1);
    mean.Memset(0);

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ComputeMeanKernel<<<blocks, threads>>>(*pcl.device_, *mean.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    auto downloaded_mean = mean.DownloadAll();
    return downloaded_mean[0].ToEigen();
}

__global__
void NormalizePointCloudKernel(PointCloudCudaDevice server,
                               Vector3f mean,
                               ArrayCudaDevice<float> scale) {
    __shared__ float local_max[THREAD_1D_UNIT];

    const int tid = threadIdx.x;
    local_max[tid] = 0;

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.points_.size()) return;

    Vector3f &vertex = server.points_[idx];
    vertex -= mean;

    local_max[tid] = vertex.norm();
    __syncthreads();

    if (tid < 128) {
        local_max[tid] = fmaxf(local_max[tid + 128], local_max[tid]);
    }
    __syncthreads();

    if (tid < 64) {
        local_max[tid] = fmaxf(local_max[tid + 64], local_max[tid]);
    }
    __syncthreads();

    if (tid < 32) {
        WarpReduceMax<float>(local_max, tid);
    }

    if (tid == 0) {
        atomicMaxf(&scale[0], local_max[0]);
    }
}

double FastGlobalRegistrationCudaKernelCaller::NormalizePointCloud(
    PointCloudCuda &pcl, Eigen::Vector3d &mean) {

    ArrayCuda<float> scale;
    scale.Create(1);
    scale.Memset(0);

    Vector3f mean_cuda;
    mean_cuda.FromEigen(mean);

    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    NormalizePointCloudKernel<<<blocks, threads>>>(
        *pcl.device_, mean_cuda, *scale.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());

    auto downloaded_scale = scale.DownloadAll();
    return downloaded_scale[0];
}

__global__
void RescalePointCloudKernel(PointCloudCudaDevice server, float scale) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.points_.size()) return;

    Vector3f &vertex = server.points_[idx];
    vertex /= scale;
}

void FastGlobalRegistrationCudaKernelCaller::
RescalePointCloud(open3d::cuda::PointCloudCuda &pcl, double scale){
    const dim3 blocks(DIV_CEILING(pcl.points_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    RescalePointCloudKernel<<<blocks, threads>>>(*pcl.device_, (float) scale);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__
void ComputeResultsAndTransformationKernel(
    FastGlobalRegistrationCudaDevice fgr) {
    __shared__ float local_sum0[THREAD_1D_UNIT];
    __shared__ float local_sum1[THREAD_1D_UNIT];
    __shared__ float local_sum2[THREAD_1D_UNIT];

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int tid = threadIdx.x;

    /** Proper initialization **/
    local_sum0[tid] = 0;
    local_sum1[tid] = 0;
    local_sum2[tid] = 0;

    if (idx >= fgr.corres_final_.size()) return;

    Vector2i &pair = fgr.corres_final_[idx];
    int source_idx = pair(0);
    int target_idx = pair(1);

    Vector6f jacobian_x, jacobian_y, jacobian_z, Jtr;
    Vector3f residual;
    float lij;
    HessianCuda<6> JtJ;
    fgr.ComputePointwiseJacobianAndResidual(
        source_idx, target_idx,
        jacobian_x, jacobian_y, jacobian_z,
        residual, lij);
    ComputeJtJAndJtr(jacobian_x, jacobian_y, jacobian_z,
        residual, JtJ, Jtr);

    float rmse = residual.dot(residual) + fgr.par_ * (1 - lij) * (1 - lij);

    /** Reduce Sum JtJ **/
#pragma unroll 1
    for (size_t i = 0; i < 21; i += 3) {
        local_sum0[tid] = JtJ(i + 0);
        local_sum1[tid] = JtJ(i + 1);
        local_sum2[tid] = JtJ(i + 2);
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
            WarpReduceSum<float>(local_sum1, tid);
            WarpReduceSum<float>(local_sum2, tid);
        }

        if (tid == 0) {
            atomicAdd(&fgr.results_.at(i + 0), local_sum0[0]);
            atomicAdd(&fgr.results_.at(i + 1), local_sum1[0]);
            atomicAdd(&fgr.results_.at(i + 2), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum Jtr **/
    const int OFFSET1 = 21;
#pragma unroll 1
    for (size_t i = 0; i < 6; i += 3) {
        local_sum0[tid] = Jtr(i + 0);
        local_sum1[tid] = Jtr(i + 1);
        local_sum2[tid] = Jtr(i + 2);
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
            local_sum1[tid] += local_sum1[tid + 128];
            local_sum2[tid] += local_sum2[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
            local_sum1[tid] += local_sum1[tid + 64];
            local_sum2[tid] += local_sum2[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
            WarpReduceSum<float>(local_sum1, tid);
            WarpReduceSum<float>(local_sum2, tid);
        }

        if (tid == 0) {
            atomicAdd(&fgr.results_.at(i + 0 + OFFSET1), local_sum0[0]);
            atomicAdd(&fgr.results_.at(i + 1 + OFFSET1), local_sum1[0]);
            atomicAdd(&fgr.results_.at(i + 2 + OFFSET1), local_sum2[0]);
        }
        __syncthreads();
    }

    /** Reduce Sum rmse **/
    const int OFFSET2 = 27;
    {
        local_sum0[tid] = rmse;
        __syncthreads();

        if (tid < 128) {
            local_sum0[tid] += local_sum0[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            local_sum0[tid] += local_sum0[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            WarpReduceSum<float>(local_sum0, tid);
        }

        if (tid == 0) {
            atomicAdd(&fgr.results_.at(0 + OFFSET2), local_sum0[0]);
        }
        __syncthreads();
    }
}

void FastGlobalRegistrationCudaKernelCaller::ComputeResultsAndTransformation(
    FastGlobalRegistrationCuda &fgr) {

    const dim3 blocks(DIV_CEILING(fgr.corres_final_.size(), THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    ComputeResultsAndTransformationKernel<<<blocks, threads>>>(
        *fgr.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

} // cuda
} // open3d