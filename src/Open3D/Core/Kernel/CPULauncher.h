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

#pragma once

#include <cassert>
#include <vector>

#include "Open3D/Core/AdvancedIndexing.h"
#include "Open3D/Core/Indexer.h"
#include "Open3D/Core/ParallelUtil.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"

namespace open3d {
namespace kernel {
namespace cpu_launcher {

template <typename scalar_t, typename func_t>
void LaunchUnaryEWKernel(const Indexer& indexer, func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

template <typename scalar_t, typename func_t>
void LaunchBinaryEWKernel(const Indexer& indexer, func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetInputPtr(1, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionKernelSerial(const Indexer& indexer,
                                 func_t element_kernel) {
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

/// Create num_threads workers to compute partial reductions and then reduce to
/// the final results. This only applies to reduction op with one output.
template <typename scalar_t, typename func_t>
void LaunchReductionKernelTwoPass(const Indexer& indexer,
                                  func_t element_kernel,
                                  scalar_t identity) {
    if (indexer.NumOutputElements() != 1) {
        utility::LogError(
                "Internal error: two-pass reduction only works for "
                "single-output reduction ops.");
    }
    int64_t num_threads = parallel_util::GetMaxThreads();
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

template <typename scalar_t, typename func_t>
void LaunchReductionParallelDim(const Indexer& indexer, func_t element_kernel) {
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(0, workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

template <typename scalar_t, typename func_t>
void LaunchAdvancedIndexerKernel(const AdvancedIndexer& indexer,
                                 func_t element_kernel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t workload_idx = 0; workload_idx < indexer.NumWorkloads();
         ++workload_idx) {
        element_kernel(indexer.GetInputPtr(workload_idx),
                       indexer.GetOutputPtr(workload_idx));
    }
}

}  // namespace cpu_launcher
}  // namespace kernel
}  // namespace open3d
