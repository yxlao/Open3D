//
// Created by wei on 1/23/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Cuda/Registration/FeatureExtractorCuda.h>
#include <Cuda/Geometry/NNCuda.h>
#include "Utils.h"

int main(int argc, char **argv) {
    using namespace open3d;
    SetVerbosityLevel(VerbosityLevel::VerboseDebug);

    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_003.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_012.ply");

    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    auto source_feature_cpu = PreprocessPointCloud(*source);
    auto target_feature_cpu = PreprocessPointCloud(*target);

    open3d::cuda::FeatureExtractorCuda source_feature_extractor;
    source_feature_extractor.Compute(
        *source, KDTreeSearchParamHybrid(0.25, 100));
    auto source_feature_cuda =
        source_feature_extractor.fpfh_features_.Download();

    open3d::cuda::FeatureExtractorCuda target_feature_extractor;
    target_feature_extractor.Compute(
        *target, KDTreeSearchParamHybrid(0.25, 100));
    auto target_feature_cuda =
        target_feature_extractor.fpfh_features_.Download();

    /** 1. Check feature extraction **/
    int valid_count = 0;
    for (int i = 0; i < source_feature_cpu->Num(); ++i) {
        float norm = (
            source_feature_cpu->data_.col(i).cast<float>()
                - source_feature_cuda.col(i)).norm();
        if (norm < 0.01f * source_feature_cpu->Dimension()) {
            valid_count++;
        }
    }
    PrintInfo("Valid features: %f (%d / %d)\n",
              (float) valid_count / source_feature_cpu->Num(),
              valid_count, source_feature_cpu->Num());

    /** 2. Check feature matching **/
    KDTreeFlann target_feature_tree(*target_feature_cpu);
    std::vector<int> correspondences_cpu;
    std::vector<int> indices(1);
    std::vector<double> dist(1);
    for (int i = 0; i < source_feature_cpu->Num(); ++i) {
        target_feature_tree.SearchKNN(
            Eigen::VectorXd(source_feature_cpu->data_.col(i)), 1,
            indices, dist);
        correspondences_cpu.push_back(indices.size() > 0 ? indices[0] : -1);
    }

    cuda::NNCuda nn;
    nn.BruteForceNN(source_feature_extractor.fpfh_features_,
                    target_feature_extractor.fpfh_features_);
    auto correspondences_cuda = nn.nn_idx_.Download();

    valid_count = 0;
    for (int i = 0; i < source_feature_cpu->Num(); ++i) {
        int correspondence_cpu = correspondences_cpu[i];
        int correspondence_cuda = correspondences_cuda(0, i);
        if (correspondence_cpu == correspondence_cuda) {
            valid_count++;
        }
    }
    PrintInfo("Valid matchings: %f (%d / %d)\n",
              (float) valid_count / source_feature_cpu->Num(),
              valid_count, source_feature_cpu->Num());

    return 0;
}