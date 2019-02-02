//
// Created by wei on 10/11/18.
//
#pragma once

#include "GeometryClasses.h"
#include "ImageCuda.h"
#include "RGBDImageCuda.h"

#include <Cuda/Common/LinearAlgebraCuda.h>
#include <Cuda/Common/TransformCuda.h>

#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Container/ArrayCuda.h>

#include <Core/Geometry/PointCloud.h>

#include <memory>

namespace open3d {
namespace cuda {
class PointCloudCudaDevice {
public:
    ArrayCudaDevice<Vector3f> points_;
    ArrayCudaDevice<Vector3f> normals_;
    ArrayCudaDevice<Vector3f> colors_;

    ArrayCudaDevice<float> radius_;
    ArrayCudaDevice<float> confidences_;
    ArrayCudaDevice<int> indices_;

public:
    VertexType type_;
    int max_points_;
};

class PointCloudCuda : public Geometry3D {
public:
    std::shared_ptr<PointCloudCudaDevice> device_ = nullptr;
    ArrayCuda<Vector3f> points_;
    ArrayCuda<Vector3f> normals_;
    ArrayCuda<Vector3f> colors_;

    /** reserved for surfels **/
    ArrayCuda<float> radius_;
    ArrayCuda<float> confidences_;
    ArrayCuda<int> indices_;

public:
    VertexType type_;
    int max_points_;

public:
    PointCloudCuda();
    PointCloudCuda(VertexType type, int max_points);
    PointCloudCuda(const PointCloudCuda &other);
    PointCloudCuda &operator=(const PointCloudCuda &other);
    ~PointCloudCuda() override;

    void Reset();
    void UpdateDevice();

    void Create(VertexType type, int max_points);
    void Release();

    bool HasPoints() const;
    bool HasNormals() const;
    bool HasColors() const;

    void Upload(PointCloud &pcl);

    void Build(RGBDImageCuda &rgbd,
               PinholeCameraIntrinsicCuda &intrinsic);
    void Build(ImageCuda<Vector1f> &depth,
               PinholeCameraIntrinsicCuda &intrinsic);
    std::shared_ptr<PointCloud> Download();

public:
    void Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    void Transform(const Eigen::Matrix4d &transformation) override;

    std::tuple<Eigen::Vector3d, double> Normalize();

    Eigen::Vector3d ComputeMean();
    double SubMeanAndGetMaxScale(Eigen::Vector3d &mean);
    void Rescale(double scale);
};

class PointCloudCudaKernelCaller {
public:
    static void GetMinBound(const PointCloudCuda &pcl,
                            ArrayCuda<Vector3f> &min_bound);
    static void GetMaxBound(const PointCloudCuda &pcl,
                            ArrayCuda<Vector3f> &max_bound);

    static void ComputeSum(PointCloudCuda &pcl, ArrayCuda<Vector3f> &sum);
    static void Normalize(PointCloudCuda &pcl, const Vector3f &mean,
                          ArrayCuda<float> &max_scale);
    static void Rescale(PointCloudCuda &pcl, float scale);

    static void Transform(PointCloudCuda &pcl, TransformCuda &transform);

    static void BuildFromRGBDImage(PointCloudCuda &server,
                                   RGBDImageCuda &rgbd,
                                   PinholeCameraIntrinsicCuda &intrinsic);

    static void BuildFromDepthImage(PointCloudCuda &server,
                                    ImageCuda<Vector1f> &depth,
                                    PinholeCameraIntrinsicCuda &intrinsic);
};

__GLOBAL__
void GetMinBoundKernel(PointCloudCudaDevice pcl,
                       ArrayCudaDevice<Vector3f> min_bound);
__GLOBAL__
void GetMaxBoundKernel(PointCloudCudaDevice pcl,
                       ArrayCudaDevice<Vector3f> max_bound);

__GLOBAL__
void ComputeSumKernel(PointCloudCudaDevice server,
                      ArrayCudaDevice<Vector3f> sum);
__GLOBAL__
void SubMeanAndGetMaxScaleKernel(PointCloudCudaDevice server,
                                 Vector3f mean,
                                 ArrayCudaDevice<float> scale);
__GLOBAL__
void RescaleKernel(PointCloudCudaDevice server, float scale);

__GLOBAL__
void TransformKernel(PointCloudCudaDevice pcl,
                     TransformCuda transform);

__GLOBAL__
void BuildFromRGBDImageKernel(PointCloudCudaDevice pcl,
                              RGBDImageCudaDevice rgbd,
                              PinholeCameraIntrinsicCuda intrinsic);

__GLOBAL__
void BuildFromDepthImageKernel(PointCloudCudaDevice pcl,
                               ImageCudaDevice<Vector1f> depth,
                               PinholeCameraIntrinsicCuda intrinsic);

} // cuda
} // open3d