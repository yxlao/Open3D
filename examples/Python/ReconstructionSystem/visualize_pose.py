from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d

if __name__ == "__main__":

    for frame_id in range(2):
        example_dir = Path(
            "/home/ylao/repo/Open3D/examples/Python/ReconstructionSystem")
        fragment_dir = example_dir / "dataset/realsense/fragments"

        pose_graph_path = fragment_dir / f"fragment_optimized_{frame_id:03d}.json"
        pose_graph = o3d.io.read_pose_graph(str(pose_graph_path))

        Ts = []
        camera_centers = []
        num_nodes = len(pose_graph.nodes)
        print("num_nodes", num_nodes)
        for i in range(num_nodes):
            pose = pose_graph.nodes[i].pose
            T = np.linalg.inv(pose)
            Ts.append(T)
            camera_center = plot3d.cameracenter_from_T(T)
            camera_centers.append(camera_center)

        # plot3d.plot_cameras(Ts, size=0.01)

        # Get point cloud
        pcd_path = fragment_dir / f"fragment_{frame_id:03d}.ply"
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        # Get poses as lineset
        camera_ls = o3d.geometry.LineSet()
        lines = [[x, x + 1] for x in range(num_nodes - 1)]
        colors = np.tile(np.array([1, 0, 0], dtype=np.float64), (num_nodes, 1))
        camera_ls.points = o3d.utility.Vector3dVector(camera_centers)
        camera_ls.lines = o3d.utility.Vector2iVector(lines)
        camera_ls.colors = o3d.utility.Vector3dVector(colors)

        # Get reference coordinates
        coord = o3d.geometry.create_mesh_coordinate_frame(size=0.1)

        # Visualize
        o3d.draw_geometries([pcd, camera_ls, coord])
