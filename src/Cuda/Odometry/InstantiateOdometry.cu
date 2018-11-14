//
// Created by wei on 10/4/18.
//

#include "SequentialRGBDOdometryCuda.h"
#include "RGBDOdometryCudaDevice.cuh"
#include "RGBDOdometryCudaKernel.cuh"
#include "Reduction2DCudaKernel.cuh"

namespace open3d {

template class RGBDOdometryCudaServer<3>;
template class RGBDOdometryCudaKernelCaller<3>;

template
float ReduceSum2D<Vector1f, float>(ImageCuda<Vector1f> &src);

template
float ReduceSum2DShuffle<Vector1f, float>(ImageCuda<Vector1f> &src);

template
float AtomicSum<Vector1f, float>(ImageCuda<Vector1f> &src);

template
int ReduceSum2D<Vector1b, int>(ImageCuda<Vector1b> &src);

template
int ReduceSum2DShuffle<Vector1b, int>(ImageCuda<Vector1b> &src);

template
int AtomicSum<Vector1b, int>(ImageCuda<Vector1b> &src);

}
