//
// Created by wei on 1/23/19.
//

//
// Created by wei on 1/21/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Core/Registration/FastGlobalRegistration.h>
#include <Core/Registration/ColoredICP.h>
#include <iostream>
#include <Cuda/Geometry/NNCuda.h>
#include <Cuda/Registration/FeatureExtractorCuda.h>
#include "ReadDataAssociation.h"

using namespace open3d;
std::shared_ptr<Feature> PreprocessPointCloud(PointCloud &pcd) {
    EstimateNormals(pcd, open3d::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = ComputeFPFHFeature(
        pcd, open3d::KDTreeSearchParamHybrid(0.25, 100));
    return pcd_fpfh;
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
    open3d::SetVerbosityLevel(VerbosityLevel::VerboseDebug);
    std::string filepath = "/home/wei/Work/data/stanford/lounge/fragments";
    auto source_origin = CreatePointCloudFromFile(
        filepath + "/fragment_003.ply");
    auto target_origin = CreatePointCloudFromFile(
        filepath + "/fragment_012.ply");

    auto source = open3d::VoxelDownSample(*source_origin, 0.05);
    auto target = open3d::VoxelDownSample(*target_origin, 0.05);

    for (int i = 0; i < 10; ++i) {
        Timer timer;
        timer.Start();
        auto source_feature = PreprocessPointCloud(*source);
        timer.Stop();
        PrintInfo("Feature extraction takes %f ms\n", timer.GetDuration());

        timer.Start();
        open3d::cuda::FeatureExtractorCuda fpfh_feature;
        fpfh_feature.Compute(*source, KDTreeSearchParamHybrid(0.25, 100));
        timer.Stop();
        PrintInfo("Feature extraction cuda takes %f ms\n", timer.GetDuration());
    }

    auto source_feature = PreprocessPointCloud(*source);
    auto target_feature = PreprocessPointCloud(*target);

    open3d::cuda::FeatureExtractorCuda source_fpfh_feature;
    source_fpfh_feature.Compute(*source, KDTreeSearchParamHybrid(0.25, 100));
    auto source_feature_cuda = source_fpfh_feature.fpfh_features_.Download();

    open3d::cuda::FeatureExtractorCuda target_fpfh_feature;
    target_fpfh_feature.Compute(*target, KDTreeSearchParamHybrid(0.25, 100));
    auto target_feature_cuda = target_fpfh_feature.fpfh_features_.Download();

    for (int i = 0; i < source_feature->Num(); ++i) {
        float norm =
            (source_feature->data_.col(i).cast<float>()
                - source_feature_cuda.col(i)).norm();
//        if (norm >= 0.5) {
//            std::cout << source_feature->data_.col(i).transpose() << std::endl;
//            std::cout << source_feature_cuda.col(i).transpose() << std::endl;
//        }
    }

    KDTreeFlann target_feature_tree(*target_feature);
    std::vector<int> correspondences;
    std::vector<int> indices(1);
    std::vector<double> dist(1);

    for (int i = 0; i < source_feature->Num(); ++i) {
        target_feature_tree.SearchKNN(
            Eigen::VectorXd(source_feature->data_.col(i)),
            1,
            indices,
            dist);
        correspondences.push_back(indices.size() > 0 ? indices[0] : -1);
    }
    open3d::cuda::NNCuda nn;
    nn.NNSearch(source_fpfh_feature.fpfh_features_,
                target_fpfh_feature.fpfh_features_);

    auto correspondences_cuda = nn.nn_idx_.Download();
    int inconsistency = 0;
    for (int i = 0; i < source_feature->Num(); ++i) {
        int correspondence_cpu = correspondences[i];
        int correspondence_cuda = correspondences_cuda(0, i);
        if (correspondence_cpu != correspondence_cuda) {
            ++inconsistency;
        }
    }
    PrintInfo("Consistency: %f (%d / %d)\n",
              1 - (float) inconsistency / source_feature->Num(),
              source_feature->Num() - inconsistency, source_feature->Num());

    return 0;
}