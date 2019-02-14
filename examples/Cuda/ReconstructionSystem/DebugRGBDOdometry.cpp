//
// Created by wei on 2/4/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Odometry/RGBDOdometryCuda.h>
#include <Cuda/Integration/ScalableTSDFVolumeCuda.h>
#include <Cuda/Integration/ScalableMeshVolumeCuda.h>

#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>

#include "ORBPoseEstimation.h"
#include "DatasetConfig.h"

using namespace open3d;

PoseGraph MakePoseGraphForFragment(int fragment_id, DatasetConfig &config) {

    cuda::RGBDOdometryCuda<3> odometry;
    odometry.SetIntrinsics(config.intrinsic_);
    odometry.SetParameters(OdometryOption({20, 10, 5},
                                          config.max_depth_diff_,
                                          config.min_depth_,
                                          config.max_depth_), 0.968f);

    cuda::RGBDImageCuda rgbd_source((float) config.max_depth_,
                                    (float) config.depth_factor_);
    cuda::RGBDImageCuda rgbd_target((float) config.max_depth_,
                                    (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int) config.color_files_.size());

    // world_to_source
    Eigen::Matrix4d trans_odometry = Eigen::Matrix4d::Identity();
    PoseGraph pose_graph;
    pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry));

    /** Add odometry and keyframe info **/
    std::vector<ORBPoseEstimation::KeyframeInfo> keyframe_infos;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(100);

    for (int s = begin; s < end - 1; ++s) {
        PrintInfo("s: %d\n", s);
        Image depth, color;

        ReadImage(config.depth_files_[s], depth);
        ReadImage(config.color_files_[s], color);

        PrintInfo("s: %d\n", s);
        rgbd_source.Upload(depth, color);

        int t = s + 1;
        ReadImage(config.depth_files_[t], depth);
        ReadImage(config.color_files_[t], color);
        rgbd_target.Upload(depth, color);

        odometry.transform_source_to_target_ = Eigen::Matrix4d::Identity();
        odometry.Initialize(rgbd_source, rgbd_target);
        odometry.ComputeMultiScale();

        Eigen::Matrix4d trans = odometry.transform_source_to_target_;
        Eigen::Matrix6d information = odometry.ComputeInformationMatrix();

        // source_to_target * world_to_source = world_to_target
        trans_odometry = trans * trans_odometry;

        // target_to_world
        Eigen::Matrix4d trans_odometry_inv = trans_odometry.inverse();

        pose_graph.nodes_.emplace_back(PoseGraphNode(trans_odometry_inv));
        pose_graph.edges_.emplace_back(PoseGraphEdge(
            s - begin, t - begin, trans, information, false));

//        std::shared_ptr<cuda::PointCloudCuda>
//            pcl_source = std::make_shared<cuda::PointCloudCuda>(
//            cuda::VertexWithColor, 300000),
//            pcl_target = std::make_shared<cuda::PointCloudCuda>(
//            cuda::VertexWithColor, 300000);
//        pcl_source->Build(odometry.source_[0],
//                          odometry.device_->intrinsics_[0]);
//        pcl_target->Build(odometry.target_[0],
//                          odometry.device_->intrinsics_[0]);
//        pcl_source->Transform(odometry.transform_source_to_target_);
//        DrawGeometries({pcl_source, pcl_target});

        /** Insert a keyframe **/
        if (config.with_opencv_ && s % config.n_keyframes_per_n_frame_ == 0) {
            cv::Mat im;
            rgbd_source.intensity_.DownloadMat().convertTo(im, CV_8U, 255.0);
            std::vector<cv::KeyPoint> kp;
            cv::Mat desc;
            orb->detectAndCompute(im, cv::noArray(), kp, desc);

            ORBPoseEstimation::KeyframeInfo keyframe_info;
            keyframe_info.idx = s;
            keyframe_info.descriptor = desc;
            keyframe_info.keypoints = kp;
            keyframe_info.depth = rgbd_source.depth_.DownloadMat();
            keyframe_info.color = im;
            keyframe_infos.emplace_back(keyframe_info);
        }
    }

    /** Add Loop closures **/
    PrintInfo("Loop closure\n");
    if (config.with_opencv_) {
        for (int i = 0; i < keyframe_infos.size() - 1; ++i) {
            for (int j = i + 1; j < keyframe_infos.size(); ++j) {
                int s = keyframe_infos[i].idx;
                int t = keyframe_infos[j].idx;
                PrintInfo("matching (%d %d)\n", s, t);

                bool is_success;
                Eigen::Matrix4d trans_source_to_target;

                std::tie(is_success, trans_source_to_target) =
                    ORBPoseEstimation::PoseEstimationPnP(keyframe_infos[i],
                                                      keyframe_infos[j],
                                                      config.intrinsic_);
                if (is_success) {
                    Image depth, color;

                    ReadImage(config.depth_files_[s], depth);
                    ReadImage(config.color_files_[s], color);
                    auto ss = CreateRGBDImageFromColorAndDepth(color, depth);

                    rgbd_source.Upload(depth, color);

                    ReadImage(config.depth_files_[t], depth);
                    ReadImage(config.color_files_[t], color);
                    auto tt = CreateRGBDImageFromColorAndDepth(color, depth);

                    rgbd_target.Upload(depth, color);

                    odometry.Initialize(rgbd_source, rgbd_target);
                    odometry.transform_source_to_target_ =
                        trans_source_to_target;
                    auto result = odometry.ComputeMultiScale();

                    if (std::get<0>(result)) {
                        Eigen::Matrix4d
                            trans = odometry.transform_source_to_target_;
                        Eigen::Matrix6d
                            information = odometry.ComputeInformationMatrix();

                        auto s_p = CreatePointCloudFromRGBDImage(*ss, config
                            .intrinsic_);
                        auto t_p = CreatePointCloudFromRGBDImage(*tt, config
                            .intrinsic_);
                        s_p->Transform(trans);
                        DrawGeometries({s_p, t_p});

                        PrintInfo("Add edge (%d %d)\n", s, t);
                        std::cout << trans << "\n" << information << "\n";
                        pose_graph.edges_.emplace_back(PoseGraphEdge(
                            s - begin, t - begin, trans, information, true));
                    }
                }
            }
        }
    }

    return pose_graph;
}


PoseGraph OptimizePoseGraphForFragment(int fragment_id, PoseGraph &pose_graph,
                                  DatasetConfig &config) {

    std::cout << config.preference_loop_closure_odometry_ << std::endl;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    GlobalOptimizationConvergenceCriteria criteria;
    GlobalOptimizationOption option(
        config.max_depth_diff_,
        0.25,
        config.preference_loop_closure_odometry_,
        0);
    GlobalOptimizationLevenbergMarquardt optimization_method;
    GlobalOptimization(pose_graph, optimization_method,
                       criteria, option);
    SetVerbosityLevel(VerbosityLevel::VerboseInfo);

    return pose_graph;
}

void IntegrateForFragment(int fragment_id, PoseGraph &pose_graph,
                          DatasetConfig &config) {

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda<8> tsdf_volume(
        20000,
        400000,
        voxel_length,
        (float) config.tsdf_truncation_,
        trans);

    cuda::RGBDImageCuda rgbd((float) config.max_depth_,
                             (float) config.depth_factor_);

    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int
        end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                       (int) config.color_files_.size());

    for (int i = begin; i < end; ++i) {
        PrintDebug("Integrating frame %d ...\n", i);

        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = pose_graph.nodes_[i - begin].pose_;
        trans.FromEigen(pose);

        tsdf_volume.Integrate(rgbd, intrinsic, trans);
    }

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda<8> mesher(
        tsdf_volume.active_subvolume_entry_array().size(),
        cuda::VertexWithNormalAndColor, 10000000, 20000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();

    PointCloud pcl;
    pcl.points_ = mesh->vertices_;
    pcl.normals_ = mesh->vertex_normals_;
    pcl.colors_ = mesh->vertex_colors_;

    std::shared_ptr<PointCloud> ptr = std::make_shared<PointCloud>(pcl);
    DrawGeometries({ptr});
    WritePointCloudToPLY("/home/wei/fragment_057_cuda_nocv.ply", pcl);
}


int main(int argc, char **argv) {
    Timer timer;
    timer.Start();

    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
                              "/home/wei/Work/projects/dense_mapping/Open3D/examples/Cuda"
                              "/ReconstructionSystem/config/apartment.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    filesystem::MakeDirectory(config.path_dataset_ + "/fragments_cuda");

    config.with_opencv_ = false;
    const int num_fragments =
        DIV_CEILING(config.color_files_.size(),
                    config.n_frames_per_fragment_);

    for (int i = 57; i < 58; ++i) {
        PrintInfo("Processing fragment %d / %d\n", i, num_fragments - 1);
        auto pose_graph = MakePoseGraphForFragment(i, config);
        WritePoseGraph(
        "/media/wei/Data/data/indoor_lidar_rgbd/apartment/fragments_cuda/"
        "fragment_057.json", pose_graph);
        auto pose_graph_prunned = OptimizePoseGraphForFragment(i, pose_graph,
            config);
        WritePoseGraph(
            "/media/wei/Data/data/indoor_lidar_rgbd/apartment/fragments_cuda/"
            "fragment_optimizedd_057.json", pose_graph_prunned);
        IntegrateForFragment(i, pose_graph_prunned, config);
    }
    timer.Stop();
    PrintInfo("MakeFragment takes %.3f s\n", timer.GetDuration() / 1000.0f);
}