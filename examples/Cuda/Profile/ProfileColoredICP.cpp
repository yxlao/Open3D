//
// Created by wei on 2/21/19.
//

#include <vector>
#include <string>
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include <Cuda/Registration/RegistrationCuda.h>
#include <Cuda/Registration/ColoredICPCuda.h>
#include <Cuda/Registration/FastGlobalRegistrationCuda.h>

#include <Core/Registration/PoseGraph.h>
#include <Core/Registration/GlobalOptimization.h>
#include <Core/Registration/ColoredICP.h>

#include "examples/Cuda/DatasetConfig.h"
#include "Analyzer.h"

using namespace open3d;

double MultiScaleICP(const PointCloud &source, const PointCloud &target,
                     const Eigen::Matrix4d &init_trans,
                     const float voxel_size,
                     bool use_cuda,
                     const std::vector<int> &iters = {50, 30, 14},
                     const std::vector<float> &voxel_factors = {1.0, 2.0,
                                                                4.0}) {

    assert(iters.size() == voxel_factors.size());

    Eigen::Matrix4d transformation = init_trans;
    Eigen::Matrix6d information = Eigen::Matrix6d::Identity();

    Timer timer;
    double time = 0;
    for (int i = 0; i < iters.size(); ++i) {
        float voxel_size_level = voxel_size / voxel_factors[i];
        auto source_down = VoxelDownSample(source, voxel_size_level);
        auto target_down = VoxelDownSample(target, voxel_size_level);

        if (use_cuda) {
            timer.Start();
            cuda::RegistrationCuda registration(
                TransformationEstimationType::ColoredICP);
            registration.Initialize(*source_down, *target_down,
                                    voxel_size_level * 1.4f,
                                    transformation);
            registration.ComputeICP(iters[i]);
            transformation = registration.transform_source_to_target_;
            timer.Stop();
            time += timer.GetDuration();
        } else {
            timer.Start();
            auto result = RegistrationColoredICP(*source_down, *target_down,
                                                 voxel_size_level * 1.4f,
                                                 transformation);
            transformation = result.transformation_;
            timer.Stop();
            time += timer.GetDuration();
        }
    }

    return time;
}

void ProfileRegistration(DatasetConfig &config, bool use_cuda) {
    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForScene(true), pose_graph);

    std::vector<double> times;

    for (auto &edge : pose_graph.edges_) {
        Match match;
        match.s = edge.source_node_id_;
        match.t = edge.target_node_id_;
        match.success = true;

        auto source = CreatePointCloudFromFile(config.fragment_files_[match.s]);
        auto target = CreatePointCloudFromFile(config.fragment_files_[match.t]);

        double time = MultiScaleICP(*source, *target,
                                    edge.transformation_,
                                    config.voxel_size_, use_cuda);
        times.push_back(time);
        PrintInfo("Fragment %d - %d takes %f ms\n", match.s, match.t, time);
    }

    double mean, std;
    std::tie(mean, std) = ComputeStatistics(times);
    PrintInfo("gpu time: avg = %f, std = %f\n", mean, std);
}

int main(int argc, char **argv) {
    DatasetConfig config;

    std::string config_path = argc > 1 ? argv[1] :
        kDefaultDatasetConfigDir + "/stanford/lounge.json";

    bool is_success = ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    is_success = config.GetFragmentFiles();
    if (!is_success) {
        PrintError("Unable to get fragment files\n");
        return -1;
    }

    ProfileRegistration(config, true);
    ProfileRegistration(config, false);
}