// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Open3D/Core/Kernel/Reduction.h"

#include <limits>

#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/ParallelUtil.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {

template <typename scalar_t>
static inline scalar_t CPUSumReductionKernel(scalar_t src, scalar_t dst) {
    return src + dst;
}

template <typename scalar_t>
static inline scalar_t CPUProdReductionKernel(scalar_t src, scalar_t dst) {
    return src * dst;
}

template <typename scalar_t>
static inline scalar_t CPUMinReductionKernel(scalar_t src, scalar_t dst) {
    return std::min(src, dst);
}

template <typename scalar_t>
static inline scalar_t CPUMaxReductionKernel(scalar_t src, scalar_t dst) {
    return std::max(src, dst);
}

class CPUReductionEngine {
public:
    CPUReductionEngine(const CPUReductionEngine&) = delete;
    CPUReductionEngine& operator=(const CPUReductionEngine&) = delete;
    CPUReductionEngine(const Indexer& indexer) : indexer_(indexer) {}

    template <typename func_t, typename scalar_t>
    void Run(const func_t& reduce_func, scalar_t identity) {
        // See: PyTorch's TensorIterator::parallel_reduce for the reference
        // design of reduction strategy.
        if (parallel_util::GetMaxThreads() == 1 ||
            parallel_util::InParallel()) {
            LaunchReductionKernelSerial<scalar_t>(indexer_, reduce_func);
        } else if (indexer_.NumOutputElements() <= 1) {
            LaunchReductionKernelTwoPass<scalar_t>(indexer_, reduce_func,
                                                   identity);
        } else {
            LaunchReductionParallelDim<scalar_t>(indexer_, reduce_func);
        }
    }

private:
    template <typename scalar_t, typename func_t>
    static void LaunchReductionKernelSerial(const Indexer& indexer,
                                            func_t element_kernel) {
        for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
             ++workload_idx) {
            scalar_t* src = reinterpret_cast<scalar_t*>(
                    indexer.GetInputPtr(0, workload_idx));
            scalar_t* dst = reinterpret_cast<scalar_t*>(
                    indexer.GetOutputPtr(workload_idx));
            *dst = element_kernel(*src, *dst);
        }
    }

    /// Create num_threads workers to compute partial reductions and then reduce
    /// to the final results. This only applies to reduction op with one output.
    template <typename scalar_t, typename func_t>
    static void LaunchReductionKernelTwoPass(const Indexer& indexer,
                                             func_t element_kernel,
                                             scalar_t identity) {
        if (indexer.NumOutputElements() > 1) {
            utility::LogError(
                    "Internal error: two-pass reduction only works for "
                    "single-output reduction ops.");
        }
        int64_t num_workloads = indexer.NumWorkloads();
        int64_t num_threads = parallel_util::GetMaxThreads();
        int64_t workload_per_thread =
                (num_workloads + num_threads - 1) / num_threads;
        std::vector<scalar_t> thread_results(num_threads, identity);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            int64_t start = thread_idx * workload_per_thread;
            int64_t end = std::min(start + workload_per_thread, num_workloads);
            for (int64_t workload_idx = start; workload_idx < end;
                 ++workload_idx) {
                scalar_t* src = reinterpret_cast<scalar_t*>(
                        indexer.GetInputPtr(0, workload_idx));
                thread_results[thread_idx] =
                        element_kernel(*src, thread_results[thread_idx]);
            }
        }
        scalar_t* dst = reinterpret_cast<scalar_t*>(indexer.GetOutputPtr(0));
        for (int64_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            *dst = element_kernel(thread_results[thread_idx], *dst);
        }
    }

    template <typename scalar_t, typename func_t>
    static void LaunchReductionParallelDim(const Indexer& indexer,
                                           func_t element_kernel) {
        // Prefers outer dimension >= num_threads.
        const int64_t* indexer_shape = indexer.GetMasterShape();
        const int64_t num_dims = indexer.NumDims();
        int64_t num_threads = parallel_util::GetMaxThreads();

        // Init best_dim as the outer-most non-reduction dim.
        int64_t best_dim = num_dims - 1;
        while (best_dim >= 0 && indexer.IsReductionDim(best_dim)) {
            best_dim--;
        }
        for (int64_t dim = best_dim; dim >= 0 && !indexer.IsReductionDim(dim);
             --dim) {
            if (indexer_shape[dim] >= num_threads) {
                best_dim = dim;
                break;
            } else if (indexer_shape[dim] > indexer_shape[best_dim]) {
                best_dim = dim;
            }
        }
        if (best_dim == -1) {
            utility::LogError(
                    "Internal error: all dims are reduction dims, use "
                    "LaunchReductionKernelTwoPass instead.");
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < indexer_shape[best_dim]; ++i) {
            Indexer sub_indexer(indexer);
            sub_indexer.ShrinkDim(best_dim, i, 1);
            LaunchReductionKernelSerial<scalar_t>(sub_indexer, element_kernel);
        }
    }

private:
    Indexer indexer_;
};

void ReductionCPU(const Tensor& src,
                  Tensor& dst,
                  const SizeVector& dims,
                  bool keepdim,
                  ReductionOpCode op_code) {
    DtypePolicy dtype_policy;
    if (regular_reduce_ops.find(op_code) != regular_reduce_ops.end()) {
        dtype_policy = DtypePolicy::ASSERT_SAME;
    } else if (arg_reduce_ops.find(op_code) != regular_reduce_ops.end()) {
        dtype_policy = DtypePolicy::ASSERT_SAME_INPUTS;
    } else {
        utility::LogError("Unsupported op code.");
    }

    Indexer indexer({src}, dst, dtype_policy, dims);
    CPUReductionEngine re(indexer);

    Dtype dtype = src.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t identity;
        std::function<scalar_t(scalar_t, scalar_t)> element_kernel;
        switch (op_code) {
            case ReductionOpCode::Sum:
                identity = 0;
                dst.Fill(identity);
                re.Run(CPUSumReductionKernel<scalar_t>, identity);
                break;
            case ReductionOpCode::Prod:
                identity = 1;
                dst.Fill(identity);
                re.Run(CPUSumReductionKernel<scalar_t>, identity);
                break;
            case ReductionOpCode::Min:
                if (indexer.NumWorkloads() == 0) {
                    utility::LogError("Zero-size Tensor does not suport Min.");
                } else {
                    identity = std::numeric_limits<scalar_t>::max();
                    dst.Fill(identity);
                    re.Run(CPUMinReductionKernel<scalar_t>, identity);
                }
                break;
            case ReductionOpCode::Max:
                if (indexer.NumWorkloads() == 0) {
                    utility::LogError("Zero-size Tensor does not suport Max.");
                } else {
                    identity = std::numeric_limits<scalar_t>::min();
                    dst.Fill(identity);
                    re.Run(CPUMaxReductionKernel<scalar_t>, identity);
                }
                break;
            case ReductionOpCode::ArgMin:
                if (indexer.NumWorkloads() == 0) {
                    utility::LogError(
                            "Zero-size Tensor does not suport ArgMin.");
                } else {
                    utility::LogError("TODO: ArgMin CPU is not implemented.");
                }
                break;
            case ReductionOpCode::ArgMax:
                if (indexer.NumWorkloads() == 0) {
                    utility::LogError(
                            "Zero-size Tensor does not suport ArgMax.");
                } else {
                    utility::LogError("TODO: ArgMax CPU is not implemented.");
                }
                break;
            default:
                utility::LogError("Unsupported op code.");
                break;
        }
    });
}

}  // namespace kernel
}  // namespace open3d
