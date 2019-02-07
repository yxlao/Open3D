//
// Created by wei on 2/5/19.
//

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include "System.h"

using namespace open3d;

int main(int argc, char **argv) {
    PoseGraph pose_graph;
    ReadPoseGraph(GetScenePoseGraphName(kBasePath, ""),
                  pose_graph);

    std::shared_ptr<LineSet> pose_graph_vis = std::make_shared<LineSet>();

    int cnt = 0;

    const int kPointsPerFrustum = 5;
    const int kEdgesPerFrustum = 8;
    for (auto &node : pose_graph.nodes_) {
        auto pose = node.pose_;

        double norm = 0.1;
        pose_graph_vis->points_.emplace_back(
            (pose * Eigen::Vector4d(0, 0, 0, 1)).hnormalized());
        pose_graph_vis->points_.emplace_back(
            (pose * (norm * Eigen::Vector4d(1, 1, 2, 1/norm))).hnormalized());
        pose_graph_vis->points_.emplace_back(
            (pose * (norm * Eigen::Vector4d(1, -1, 2, 1/norm))).hnormalized());
        pose_graph_vis->points_.emplace_back(
            (pose * (norm * Eigen::Vector4d(-1, -1, 2, 1/norm))).hnormalized());
        pose_graph_vis->points_.emplace_back(
            (pose * (norm * Eigen::Vector4d(-1, 1, 2, 1/norm))).hnormalized());

        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 0, cnt + 1));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 0, cnt + 2));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 0, cnt + 3));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 0, cnt + 4));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 1, cnt + 2));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 2, cnt + 3));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 3, cnt + 4));
        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(cnt + 4, cnt + 1));

        for (int k = 0; k < kEdgesPerFrustum; ++k) {
            pose_graph_vis->colors_.emplace_back(Eigen::Vector3d(1, 0, 0));
        }

        cnt += kPointsPerFrustum;
    }

    for (auto &edge : pose_graph.edges_) {
        int s = edge.source_node_id_;
        int t = edge.target_node_id_;

        pose_graph_vis->lines_.emplace_back(
            Eigen::Vector2i(s * kPointsPerFrustum, t * kPointsPerFrustum));
        if (edge.uncertain_) {
            pose_graph_vis->colors_.emplace_back(Eigen::Vector3d(0, 1, 0));
        } else {
            pose_graph_vis->colors_.emplace_back(Eigen::Vector3d(0, 0, 1));
        }
    }

    DrawGeometries({pose_graph_vis});

    return 0;
}