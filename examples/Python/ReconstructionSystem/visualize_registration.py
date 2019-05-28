# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/non_blocking_visualization.py

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details


from pathlib import Path
import open3d as o3d
import numpy as np



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
    source, _ = preprocess_point_cloud(source_raw, voxel_size=voxel_size)
    target, _ = preprocess_point_cloud(target_raw, voxel_size=voxel_size)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    icp_iteration = 50
    save_image = False

    for i in range(icp_iteration):
        result_icp = o3d.registration.registration_colored_icp(
            source, target, threshold, np.identity(4),
            o3d.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=1)
        )
        # time.sleep(10)
        source.transform(result_icp.transformation)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
