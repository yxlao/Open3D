//
// Created by wei on 10/11/18.
//
#pragma once

#include "GeometryClasses.h"
#include "ImageCuda.h"
#include "RGBDImageCuda.h"

#include <Cuda/Common/VectorCuda.h>
#include <Cuda/Common/TransformCuda.h>

#include <Cuda/Camera/PinholeCameraIntrinsicCuda.h>
#include <Cuda/Container/ArrayCuda.h>

#include <Core/Geometry/PointCloud.h>

#include <memory>

namespace open3d {
namespace cuda {
class PointCloudCudaServer {
private:
    ArrayCudaServer<Vector3f> points_;
    ArrayCudaServer<Vector3f> normals_;
    ArrayCudaServer<Vector3f> colors_;

    ArrayCudaServer<float> radius_;
    ArrayCudaServer<float> confidences_;
    ArrayCudaServer<int> indices_;

public:
    VertexType type_;
    int max_points_;

public:
    __HOSTDEVICE__ inline ArrayCudaServer<Vector3f> &points() {
        return points_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<Vector3f> &normals() {
        return normals_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<Vector3f> &colors() {
        return colors_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<float> &radius() {
        return radius_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<float> &confidences() {
        return confidences_;
    }
    __HOSTDEVICE__ inline ArrayCudaServer<int> &indices() {
        return indices_;
    }

public:
    friend class PointCloudCuda;
};

class PointCloudCuda : public Geometry3D {
private:
    std::shared_ptr<PointCloudCudaServer> server_ = nullptr;
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
    void UpdateServer();

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

public:
    ArrayCuda<Vector3f> &points() { return points_; }
    const ArrayCuda<Vector3f> &points() const { return points_; }
    ArrayCuda<Vector3f> &normals() { return normals_; }
    const ArrayCuda<Vector3f> &normals() const { return normals_; }
    ArrayCuda<Vector3f> &colors() { return colors_; }
    const ArrayCuda<Vector3f> &colors() const { return colors_; }
    ArrayCuda<float> &radius() { return radius_; }
    const ArrayCuda<float> &radius() const { return radius_; }
    ArrayCuda<float> &confidences() { return confidences_; }
    const ArrayCuda<float> &confidences() const { return confidences_; }
    ArrayCuda<int> &indices() { return indices_; }
    const ArrayCuda<int> &indices() const { return indices_; }

    std::shared_ptr<PointCloudCudaServer> &server() {
        return server_;
    }
    const std::shared_ptr<PointCloudCudaServer> &server() const {
        return server_;
    }
};

class PointCloudCudaKernelCaller {
public:
    static __HOST__ void GetMinBoundKernelCaller(
        PointCloudCudaServer &server,
        ArrayCudaServer<Vector3f> &min_bound,
        int num_vertices);

    static __HOST__ void GetMaxBoundKernelCaller(
        PointCloudCudaServer &server,
        ArrayCudaServer<Vector3f> &max_bound,
        int num_vertices);

    static __HOST__ void TransformKernelCaller(
        PointCloudCudaServer &server,
        TransformCuda &transform,
        int num_vertices);

    static __HOST__ void BuildFromRGBDImageKernelCaller(
        PointCloudCudaServer &server,
        RGBDImageCudaServer &rgbd,
        PinholeCameraIntrinsicCuda &intrinsic);

    static __HOST__ void BuildFromDepthImageKernelCaller(
        PointCloudCudaServer &server,
        ImageCudaServer<Vector1f> &depth,
        PinholeCameraIntrinsicCuda &intrinsic);
};

__GLOBAL__
void BuildFromRGBDImageKernel(PointCloudCudaServer server,
                              RGBDImageCudaServer rgbd,
                              PinholeCameraIntrinsicCuda intrinsic);

__GLOBAL__
void BuildFromDepthImageKernel(PointCloudCudaServer server,
                               ImageCudaServer<Vector1f> depth,
                               PinholeCameraIntrinsicCuda intrinsic);

__GLOBAL__
void GetMinBoundKernel(PointCloudCudaServer server,
                       ArrayCudaServer<Vector3f> min_bound);

__GLOBAL__
void GetMaxBoundKernel(PointCloudCudaServer server,
                       ArrayCudaServer<Vector3f> max_bound);

__GLOBAL__
void TransformKernel(PointCloudCudaServer, TransformCuda transform);
} // cuda
} // open3d