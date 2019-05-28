from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d


def get_camera_line_set(T, size=0.1, color=np.array([0, 0, 1])):

    def cameracenter_from_translation(R, t):
        # - R.T @ t
        t = t.reshape(-1, 3, 1)
        R = R.reshape(-1, 3, 3)
        C = -R.transpose(0, 2, 1) @ t
        return C.squeeze()

    R, t = T[:3, :3], T[:3, 3]

    C0 = cameracenter_from_translation(R, t).ravel()
    C1 = (C0 + R.T.dot(
        np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel())
    C2 = (C0 + R.T.dot(
        np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C3 = (C0 + R.T.dot(
        np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C4 = (C0 + R.T.dot(
        np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel())

    ls = o3d.geometry.LineSet()
    points = np.array([C0, C1, C2, C3, C4])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


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
        colors = np.tile(np.array([1, 0, 0], dtype=np.float64),
                         (num_nodes - 1, 1))
        camera_ls.points = o3d.utility.Vector3dVector(camera_centers)
        camera_ls.lines = o3d.utility.Vector2iVector(lines)
        camera_ls.colors = o3d.utility.Vector3dVector(colors)

        # Get reference coordinates
        camera_frames = o3d.geometry.LineSet()
        for T in Ts:
            camera_frame = get_camera_line_set(T, size=0.02)
            camera_frames += camera_frame

        # Visualize
        o3d.draw_geometries([pcd, camera_ls, camera_frames])
