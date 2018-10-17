//
// Created by wei on 10/10/18.
//

#include "UniformTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCudaKernel.cuh"

#include "UniformMeshVolumeCuda.cuh"

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
int UniformTSDFVolumeCuda<8>::MarchingCubes<VertexRaw>(
    UniformMeshVolumeCuda<VertexRaw, 8> &mesher);
template
int UniformTSDFVolumeCuda<8>::MarchingCubes<VertexWithNormal>(
    UniformMeshVolumeCuda<VertexWithNormal, 8> &mesher);
template
int UniformTSDFVolumeCuda<8>::MarchingCubes<VertexWithColor>(
    UniformMeshVolumeCuda<VertexWithColor, 8> &mesher);
template
int UniformTSDFVolumeCuda<8>::MarchingCubes<VertexWithNormalAndColor>(
    UniformMeshVolumeCuda<VertexWithNormalAndColor, 8> &mesher);

template
int UniformTSDFVolumeCuda<16>::MarchingCubes<VertexRaw>(
    UniformMeshVolumeCuda<VertexRaw, 16> &mesher);
template
int UniformTSDFVolumeCuda<16>::MarchingCubes<VertexWithNormal>(
    UniformMeshVolumeCuda<VertexWithNormal, 16> &mesher);
template
int UniformTSDFVolumeCuda<16>::MarchingCubes<VertexWithColor>(
    UniformMeshVolumeCuda<VertexWithColor, 16> &mesher);
template
int UniformTSDFVolumeCuda<16>::MarchingCubes<VertexWithNormalAndColor>(
    UniformMeshVolumeCuda<VertexWithNormalAndColor, 16> &mesher);

template
int UniformTSDFVolumeCuda<256>::MarchingCubes<VertexRaw>(
    UniformMeshVolumeCuda<VertexRaw, 256> &mesher);
template
int UniformTSDFVolumeCuda<256>::MarchingCubes<VertexWithNormal>(
    UniformMeshVolumeCuda<VertexWithNormal, 256> &mesher);
template
int UniformTSDFVolumeCuda<256>::MarchingCubes<VertexWithColor>(
    UniformMeshVolumeCuda<VertexWithColor, 256> &mesher);
template
int UniformTSDFVolumeCuda<256>::MarchingCubes<VertexWithNormalAndColor>(
    UniformMeshVolumeCuda<VertexWithNormalAndColor, 256> &mesher);

template
int UniformTSDFVolumeCuda<512>::MarchingCubes<VertexRaw>(
    UniformMeshVolumeCuda<VertexRaw, 512> &mesher);
template
int UniformTSDFVolumeCuda<512>::MarchingCubes<VertexWithNormal>(
    UniformMeshVolumeCuda<VertexWithNormal, 512> &mesher);
template
int UniformTSDFVolumeCuda<512>::MarchingCubes<VertexWithColor>(
    UniformMeshVolumeCuda<VertexWithColor, 512> &mesher);
template
int UniformTSDFVolumeCuda<512>::MarchingCubes<VertexWithNormalAndColor>(
    UniformMeshVolumeCuda<VertexWithNormalAndColor, 512> &mesher);

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

}