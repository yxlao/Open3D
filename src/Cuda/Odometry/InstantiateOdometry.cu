//
// Created by wei on 10/4/18.
//

#include "RGBDOdometryCuda.h"
#include "RGBDOdometryCudaDevice.cuh"
#include "RGBDOdometryCudaKernel.cuh"

#include "ICRGBDOdometryCuda.h"
#include "ICRGBDOdometryCudaDevice.cuh"
#include "ICRGBDOdometryCudaKernel.cuh"

#include "Reduction2DCudaKernel.cuh"

#include <src/Cuda/Geometry/ImageCudaDevice.cuh>
#include <src/Cuda/Geometry/ImageCudaKernel.cuh>
#include <src/Cuda/Geometry/ImagePyramidCuda.h>

namespace open3d {
namespace cuda {
template
class RGBDOdometryCudaDevice<3>;
template
class RGBDOdometryCudaKernelCaller<3>;

template
class ICRGBDOdometryCudaDevice<3>;
template
class ICRGBDOdometryCudaKernelCaller<3>;

template
class ImageCudaDevice<Vector6f>;
template
class ImageCudaKernelCaller<Vector6f>;
template
class ImagePyramidCudaDevice<Vector6f, 3>;

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

} // cuda
} // open3d