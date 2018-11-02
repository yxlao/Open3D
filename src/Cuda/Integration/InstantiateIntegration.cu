//
// Created by wei on 10/10/18.
//

#include "UniformTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCudaKernel.cuh"
#include "UniformMeshVolumeCuda.cuh"
#include "UniformMeshVolumeCudaKernel.cuh"
#include "ScalableTSDFVolumeCuda.cuh"
#include "ScalableTSDFVolumeCudaKernel.cuh"
#include "ScalableMeshVolumeCuda.cuh"
#include "ScalableMeshVolumeCudaKernel.cuh"

namespace open3d {

template
class UniformTSDFVolumeCudaServer<8>;
template
class UniformTSDFVolumeCudaServer<16>;
template
class UniformTSDFVolumeCudaServer<256>;
template
class UniformTSDFVolumeCudaServer<512>;

template
class UniformTSDFVolumeCuda<8>;
template
class UniformTSDFVolumeCuda<16>;
template
class UniformTSDFVolumeCuda<256>;
template
class UniformTSDFVolumeCuda<512>;

template
class UniformMeshVolumeCuda<8>;
template
class UniformMeshVolumeCuda<16>;
template
class UniformMeshVolumeCuda<256>;
template
class UniformMeshVolumeCuda<512>;


/** Scalable part **/
/** Oh we can't afford larger chunks **/
template
class ScalableTSDFVolumeCudaServer<8>;

template
class ScalableTSDFVolumeCuda<8>;

template
class ScalableMeshVolumeCuda<8>;

}
