//
// Created by wei on 1/3/19.
//

#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Core/Utility/Timer.h>
#include <Core/Registration/ColoredICP.h>
#include <Cuda/Registration/ColoredICPCuda.h>

#include "ReadDataAssociation.h"

std::vector<int> GetCorrespondences(
    open3d::PointCloud &source, open3d::PointCloud &target) {
    open3d::Timer timer;
    std::vector<int> correspondence_set;
    correspondence_set.resize(source.points_.size());

    timer.Start();
    open3d::KDTreeFlann target_kdtree;
    target_kdtree.SetGeometry(target);
    timer.Stop();
    open3d::PrintInfo("build kd tree: %f ms\n", timer.GetDuration());

    timer.Start();
#ifdef _OPENMP
    {
#endif
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < (int) source.points_.size(); i++) {
            std::vector<int> indices(1);
            std::vector<double> dists(1);
            const auto &point = source.points_[i];
            if (target_kdtree.SearchHybrid(point, 0.05, 1, indices, dists)
                > 0) {
                correspondence_set[i] = indices[0];
            }
        }
#ifdef _OPENMP
    }
#endif

    timer.Stop();
    open3d::PrintInfo("query: %f ms for %d queries\n",
                      timer.GetDuration(), source.points_.size());
    return correspondence_set;
}

std::shared_ptr<open3d::RGBDImage> ReadRGBDImage(
    const char *color_filename, const char *depth_filename,
    const open3d::PinholeCameraIntrinsic &intrinsic,
    bool visualize) {
    open3d::Image color, depth;
    ReadImage(color_filename, color);
    ReadImage(depth_filename, depth);
    open3d::PrintDebug("Reading RGBD image : \n");
    open3d::PrintDebug("     Color : %d x %d x %d (%d bits per channel)\n",
                       color.width_, color.height_,
                       color.num_of_channels_, color.bytes_per_channel_ * 8);
    open3d::PrintDebug("     Depth : %d x %d x %d (%d bits per channel)\n",
                       depth.width_, depth.height_,
                       depth.num_of_channels_, depth.bytes_per_channel_ * 8);
    double depth_scale = 1000.0, depth_trunc = 4.0;
    bool convert_rgb_to_intensity = true;
    std::shared_ptr<open3d::RGBDImage> rgbd_image =
        CreateRGBDImageFromColorAndDepth(color,
                                         depth,
                                         depth_scale,
                                         depth_trunc,
                                         convert_rgb_to_intensity);
    if (visualize) {
        auto pcd = CreatePointCloudFromRGBDImage(*rgbd_image, intrinsic);
        open3d::DrawGeometries({pcd});
    }
    return rgbd_image;
}

void VisualizeRegistration(const open3d::PointCloud &source,
                           const open3d::PointCloud &target, const Eigen::Matrix4d &Transformation)
{
    using namespace open3d;
    std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
    std::shared_ptr<PointCloud> target_ptr(new PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    DrawGeometries({ source_transformed_ptr, target_ptr }, "Registration result");
}

int main(int argc, char **argv) {
    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "data_association.txt");

    open3d::Image source_color, source_depth, target_color, target_depth;
    open3d::PinholeCameraIntrinsic intrinsic = open3d::PinholeCameraIntrinsic(
        open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto rgbd_source = ReadRGBDImage(
        (base_path + "/" + rgbd_filenames[0].second).c_str(),
        (base_path + "/" + rgbd_filenames[0].first).c_str(),
        intrinsic,
        false);
    auto rgbd_target = ReadRGBDImage(
        (base_path + "/" + rgbd_filenames[7].second).c_str(),
        (base_path + "/" + rgbd_filenames[7].first).c_str(),
        intrinsic,
        false);
    auto source_origin = CreatePointCloudFromRGBDImage(*rgbd_source, intrinsic);
    auto target_origin = CreatePointCloudFromRGBDImage(*rgbd_target, intrinsic);
    open3d::EstimateNormals(*source_origin);
    open3d::EstimateNormals(*target_origin);

    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    open3d::KDTreeFlann kdtree;
    kdtree.SetGeometry(*target);

    open3d::cuda::TransformationEstimationCudaForColoredICP colored_icp;
    open3d::Timer timer;
    timer.Start();
    colored_icp.InitializeColorGradients(*target, kdtree,
        open3d::KDTreeSearchParamHybrid(0.07 * 2.0, 30));
    timer.Stop();
    open3d::PrintInfo("Query color nn takes %f ms", timer.GetDuration());

//    auto registration_result = open3d::RegistrationColoredICP(
//        *source, *target, 0.07,
//        Eigen::Matrix4d::Identity(),
//        open3d::ICPConvergenceCriteria(),
//        0.968);
//    VisualizeRegistration(*source, *target, registration_result.transformation_);
//    return 0;
}