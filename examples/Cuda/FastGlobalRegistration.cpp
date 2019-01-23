//
// Created by wei on 1/21/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Core/Registration/FastGlobalRegistration.h>
#include <Core/Registration/ColoredICP.h>
#include <iostream>
#include "ReadDataAssociation.h"

using namespace open3d;
std::shared_ptr<Feature> PreprocessPointCloud(PointCloud &pcd)
{
    EstimateNormals(pcd, open3d::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = ComputeFPFHFeature(
        pcd, open3d::KDTreeSearchParamHybrid(0.25, 100));
    return pcd_fpfh;
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
                           const open3d::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    using namespace open3d;
    std::shared_ptr<PointCloud> source_transformed_ptr(new PointCloud);
    std::shared_ptr<PointCloud> target_ptr(new PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    DrawGeometries({source_transformed_ptr, target_ptr}, "Registration result");
}

int main(int argc, char **argv) {
    open3d::SetVerbosityLevel(open3d::VerbosityLevel::VerboseDebug);

    std::string base_path = "/home/wei/Work/data/stanford/lounge/";
    auto rgbd_filenames = ReadDataAssociation(
        base_path + "data_association.txt");

    open3d::Image source_color, source_depth, target_color, target_depth;
    open3d::PinholeCameraIntrinsic intrinsic = open3d::PinholeCameraIntrinsic(
        open3d::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    auto rgbd_source = ReadRGBDImage(
        (base_path + "/" + rgbd_filenames[3].second).c_str(),
        (base_path + "/" + rgbd_filenames[3].first).c_str(),
        intrinsic,
        false);
    auto rgbd_target = ReadRGBDImage(
        (base_path + "/" + rgbd_filenames[18].second).c_str(),
        (base_path + "/" + rgbd_filenames[18].first).c_str(),
        intrinsic,
        false);

    auto source_origin = CreatePointCloudFromRGBDImage(*rgbd_source, intrinsic);
    auto target_origin = CreatePointCloudFromRGBDImage(*rgbd_target, intrinsic);
    open3d::EstimateNormals(*source_origin);
    open3d::EstimateNormals(*target_origin);

    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    source_origin = CreatePointCloudFromFile(filepath + "/fragment_000.ply");
    target_origin = CreatePointCloudFromFile(filepath + "/fragment_003.ply");

    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    Timer timer;
    timer.Start();
    auto source_feature = PreprocessPointCloud(*source);
    auto target_feature = PreprocessPointCloud(*target);
    timer.Stop();
    PrintInfo("Feature extraction takes %f ms\n", timer.GetDuration());

    VisualizeRegistration(*source, *target, Eigen::Matrix4d::Identity());
    auto registration_result = open3d::FastGlobalRegistration(*source, *target,
        *source_feature, *target_feature);
    registration_result = open3d::RegistrationColoredICP(*source, *target, 0.07,
        registration_result.transformation_);
    VisualizeRegistration(*source, *target, registration_result.transformation_);
}