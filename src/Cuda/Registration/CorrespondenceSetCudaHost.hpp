//
// Created by wei on 1/14/19.
//

#include "CorrespondenceSetCuda.h"

namespace open3d {
namespace cuda {

void CorrespondenceSetCuda::Compress() {
    assert(corres_matrix_.max_rows_ > 0);

    CorrespondenceSetCudaKernelCaller::
    CompressCorrespondenceKernelCaller(corres_matrix_, corres_indices_);
}
}
}