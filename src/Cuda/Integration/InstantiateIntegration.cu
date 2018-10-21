//
// Created by wei on 10/10/18.
//

#include "UniformTSDFVolumeCuda.cuh"
#include "UniformTSDFVolumeCudaKernel.cuh"
#include "UniformMeshVolumeCuda.cuh"
#include "UniformMeshVolumeCudaKernel.cuh"
#include "ScalableTSDFVolumeCuda.cuh"
#include "ScalableTSDFVolumeCudaKernel.cuh"

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

/** Vertex Allocation **/
template
__global__
void MarchingCubesVertexAllocationKernel<VertexRaw, 8>(
    UniformMeshVolumeCudaServer<VertexRaw, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormal, 8>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithColor, 8>(
    UniformMeshVolumeCudaServer<VertexWithColor, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormalAndColor, 8>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);

template
__global__
void MarchingCubesVertexAllocationKernel<VertexRaw, 16>(
    UniformMeshVolumeCudaServer<VertexRaw, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormal, 16>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithColor, 16>(
    UniformMeshVolumeCudaServer<VertexWithColor, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormalAndColor, 16>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);

template
__global__
void MarchingCubesVertexAllocationKernel<VertexRaw, 256>(
    UniformMeshVolumeCudaServer<VertexRaw, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormal, 256>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithColor, 256>(
    UniformMeshVolumeCudaServer<VertexWithColor, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormalAndColor, 256>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);

template
__global__
void MarchingCubesVertexAllocationKernel<VertexRaw, 512>(
    UniformMeshVolumeCudaServer<VertexRaw, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormal, 512>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithColor, 512>(
    UniformMeshVolumeCudaServer<VertexWithColor, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexAllocationKernel<VertexWithNormalAndColor, 512>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);

/** Vertex Extraction **/
template
__global__
void MarchingCubesVertexExtractionKernel<VertexRaw, 8>(
    UniformMeshVolumeCudaServer<VertexRaw, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);

template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormal, 8>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithColor, 8>(
    UniformMeshVolumeCudaServer<VertexWithColor, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormalAndColor, 8>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 8> mesher,
    UniformTSDFVolumeCudaServer<8> tsdf_volume);

template
__global__
void MarchingCubesVertexExtractionKernel<VertexRaw, 16>(
    UniformMeshVolumeCudaServer<VertexRaw, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormal, 16>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithColor, 16>(
    UniformMeshVolumeCudaServer<VertexWithColor, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormalAndColor, 16>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 16> mesher,
    UniformTSDFVolumeCudaServer<16> tsdf_volume);

template
__global__
void MarchingCubesVertexExtractionKernel<VertexRaw, 256>(
    UniformMeshVolumeCudaServer<VertexRaw, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormal, 256>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithColor, 256>(
    UniformMeshVolumeCudaServer<VertexWithColor, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormalAndColor, 256>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 256> mesher,
    UniformTSDFVolumeCudaServer<256> tsdf_volume);

template
__global__
void MarchingCubesVertexExtractionKernel<VertexRaw, 512>(
    UniformMeshVolumeCudaServer<VertexRaw, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormal, 512>(
    UniformMeshVolumeCudaServer<VertexWithNormal, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithColor, 512>(
    UniformMeshVolumeCudaServer<VertexWithColor, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);
template
__global__
void MarchingCubesVertexExtractionKernel<VertexWithNormalAndColor, 512>(
    UniformMeshVolumeCudaServer<VertexWithNormalAndColor, 512> mesher,
    UniformTSDFVolumeCudaServer<512> tsdf_volume);

/** Triangle Extraction **/
/** Should be self contained and don't need to be instantiated? **/


/** Scalable part **/
/** Oh we can't afford larger chunks **/
template
class ScalableTSDFVolumeCudaServer<8>;

template
class ScalableTSDFVolumeCuda<8>;

template
__global__
void CreateScalableTSDFVolumesKernel<8>(ScalableTSDFVolumeCudaServer<8> server);

}