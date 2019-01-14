//
// Created by wei on 1/14/19.
//

#pragma once

#include <Cuda/Container/MatrixCuda.h>

namespace open3d {
namespace cuda {

class CorrespondenceSetCudaDevice {

};

class CorrespondenceSetCuda {
private:
    MatrixCuda<int> corres_origin_;
    MatrixCuda<int> corres_compressed_;

public:
    void Compress();

};

class CorrespondenceSetCudaKernelCaller {

};
}
}


