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
class UniformMeshVolumeCuda<VertexRaw, 8>;
template
class UniformMeshVolumeCuda<VertexRaw, 16>;
template
class UniformMeshVolumeCuda<VertexRaw, 256>;
template
class UniformMeshVolumeCuda<VertexRaw, 512>;

template
class UniformMeshVolumeCuda<VertexWithNormal, 8>;
template
class UniformMeshVolumeCuda<VertexWithNormal, 16>;
template
class UniformMeshVolumeCuda<VertexWithNormal, 256>;
template
class UniformMeshVolumeCuda<VertexWithNormal, 512>;

template
class UniformMeshVolumeCuda<VertexWithColor, 8>;
template
class UniformMeshVolumeCuda<VertexWithColor, 16>;
template
class UniformMeshVolumeCuda<VertexWithColor, 256>;
template
class UniformMeshVolumeCuda<VertexWithColor, 512>;

template
class UniformMeshVolumeCuda<VertexWithNormalAndColor, 8>;
template
class UniformMeshVolumeCuda<VertexWithNormalAndColor, 16>;
template
class UniformMeshVolumeCuda<VertexWithNormalAndColor, 256>;
template
class UniformMeshVolumeCuda<VertexWithNormalAndColor, 512>;

/** Scalable part **/
/** Oh we can't afford larger chunks **/
template
class ScalableTSDFVolumeCudaServer<8>;

template
class ScalableTSDFVolumeCuda<8>;

template
class ScalableMeshVolumeCuda<VertexRaw, 8>;

template
class ScalableMeshVolumeCuda<VertexWithNormal, 8>;

template
class ScalableMeshVolumeCuda<VertexWithColor, 8>;

template
class ScalableMeshVolumeCuda<VertexWithNormalAndColor, 8>;
}
