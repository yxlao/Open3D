//
// Created by wei on 10/4/18.
//

#include "RGBDOdometryCuda.h"
#include "RGBDOdometryCuda.cuh"
#include "RGBDOdometryCudaKernel.cuh"
#include "Reduction2DCudaKernel.cuh"

namespace three {

template
class RGBDOdometryCudaServer<3>;

template
class RGBDOdometryCuda<3>;

template
__global__
void ApplyRGBDOdometryKernel<3>(RGBDOdometryCudaServer<3> odometry,
								size_t level);


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
