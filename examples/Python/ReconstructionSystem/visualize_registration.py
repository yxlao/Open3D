# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/non_blocking_visualization.py

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/ReconstructionSystem/register_fragments.py

import numpy as np
import open3d as o3d
import sys
sys.path.append("../Utility")
from file import join, get_file_list
sys.path.append(".")
import open3d as o3d
import numpy as np
import copy
from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d
import cv2
import os
import json
import argparse
import time, datetime
import sys
sys.path.append("../Utility")
from file import check_folder_structure
sys.path.append(".")
from initialize_config import initialize_config
import time


def update_posegrph_for_scene(s, t, transformation, information, odometry,
                              pose_graph):
    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(s,
                                           t,
                                           transformation,
                                           information,
                                           uncertain=False))
    else:  # loop closure case
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(s,
                                           t,
                                           transformation,
                                           information,
                                           uncertain=True))
    return (odometry, pose_graph)


def multiscale_icp(source,
                   target,
                   voxel_size,
                   max_iter,
                   config,
                   init_transformation=np.identity(4)):
    current_transformation = init_transformation

    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_copy)
    vis.add_geometry(target_copy)

    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = config["voxel_size"] * 1.4
        print("voxel_size %f" % voxel_size[scale])
        source_down = o3d.geometry.voxel_down_sample(source, voxel_size[scale])
        target_down = o3d.geometry.voxel_down_sample(target, voxel_size[scale])
        if config["icp_method"] == "point_to_point":
            result_icp = o3d.registration.registration_icp(
                source_down, target_down, distance_threshold,
                current_transformation,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=iter))
        else:
            o3d.geometry.estimate_normals(
                source_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            o3d.geometry.estimate_normals(
                target_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] *
                                                     2.0,
                                                     max_nn=30))
            if config["icp_method"] == "point_to_plane":
                result_icp = o3d.registration.registration_icp(
                    source_down, target_down, distance_threshold,
                    current_transformation,
                    o3d.registration.TransformationEstimationPointToPlane(),
                    o3d.registration.ICPConvergenceCriteria(max_iteration=iter))
            if config["icp_method"] == "color":
                result_icp = o3d.registration.registration_colored_icp(
                    source_down, target_down, voxel_size[scale],
                    current_transformation,
                    o3d.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6,
                        relative_rmse=1e-6,
                        max_iteration=iter))
        current_transformation = result_icp.transformation
        if i == len(max_iter) - 1:
            information_matrix = o3d.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, voxel_size[scale] * 1.4,
                result_icp.transformation)

        source_copy.points = source.points
        source_copy.colors = source.colors
        source_copy.normals = source.normals
        source_copy.transform(current_transformation)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1)

    vis.destroy_window()

    return (result_icp.transformation, information_matrix)


def local_refinement(source, target, transformation_init, config):
    voxel_size = config["voxel_size"]
    (transformation, information) = \
            multiscale_icp(
            source, target,
            [voxel_size, voxel_size/2.0, voxel_size/4.0], [50, 30, 14],
            config, transformation_init)
    return (transformation, information)


def register_point_cloud_pair(ply_file_names, s, t, transformation_init,
                              config):
    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])
    (transformation, information) = \
            local_refinement(source, target, transformation_init, config)
    return (transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t, trans):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans
        self.infomation = np.identity(6)


def make_posegraph_for_refined_scene(ply_file_names, config):
    pose_graph = o3d.io.read_pose_graph(
        join(config["path_dataset"],
             config["template_global_posegraph_optimized"]))

    n_files = len(ply_file_names)
    matching_results = {}
    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id
        matching_results[s * n_files + t] = \
                matching_result(s, t, edge.transformation)

    for r in matching_results:
        (matching_results[r].transformation,
                matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names,
                matching_results[r].s, matching_results[r].t,
                matching_results[r].transformation, config)

    pose_graph_new = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph_new.nodes.append(o3d.registration.PoseGraphNode(odometry))
    for r in matching_results:
        (odometry, pose_graph_new) = update_posegrph_for_scene(
            matching_results[r].s, matching_results[r].t,
            matching_results[r].transformation, matching_results[r].information,
            odometry, pose_graph_new)
    print(pose_graph_new)


def run(config):
    print("refine rough registration of fragments.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")
    make_posegraph_for_refined_scene(ply_file_names, config)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


if __name__ == "__main__":

    example_dir = Path(
        "/home/ylao/repo/Open3D/examples/Python/ReconstructionSystem")
    dataset_dir = example_dir / "dataset" / "realsense"
    fragments_dir = dataset_dir / "fragments"

    source_path = fragments_dir / "fragment_000.ply"
    target_path = fragments_dir / "fragment_001.ply"

    voxel_size = 0.02
    threshold = 10
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    source_raw = o3d.io.read_point_cloud(str(source_path))
    target_raw = o3d.io.read_point_cloud(str(target_path))
    source, source_fpfh = preprocess_point_cloud(source_raw,
                                                 voxel_size=voxel_size)
    target, target_fpfh = preprocess_point_cloud(target_raw,
                                                 voxel_size=voxel_size)
    # source = o3d.geometry.voxel_down_sample(source_raw, voxel_size=0.02)
    # target = o3d.geometry.voxel_down_sample(target_raw, voxel_size=0.02)

    result = o3d.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=threshold))
    init_trans = result.transformation

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    icp_iteration = 1000
    save_image = False

    for i in range(icp_iteration):
        # result_icp = o3d.registration.registration_icp(
        #     source, target, threshold, np.identity(4),
        #     o3d.registration.TransformationEstimationPointToPoint(),
        #     o3d.registration.ICPConvergenceCriteria(max_iteration=1))
        result_icp = o3d.registration.registration_colored_icp(
            source, target, threshold, np.identity(4),
            o3d.registration.ICPConvergenceCriteria())
        # time.sleep(10)
        source.transform(result_icp.transformation)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
