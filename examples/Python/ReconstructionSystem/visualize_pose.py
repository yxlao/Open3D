from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d


def get_camera_frame(T, size, color):

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


def get_camera_frames(Ts, size=0.1, color=np.array([0, 0, 1])):
    camera_frames = o3d.geometry.LineSet()
    for T in Ts:
        camera_frame = get_camera_frame(T, size=size, color=color)
        camera_frames += camera_frame
    return camera_frames


def get_camera_centers_lineset(Ts, color=np.array([1, 0, 0])):
    num_nodes = len(Ts)
    camera_centers = [plot3d.cameracenter_from_T(T) for T in Ts]

    ls = o3d.geometry.LineSet()
    lines = [[x, x + 1] for x in range(num_nodes - 1)]
    colors = np.tile(color, (len(lines), 1))
    ls.points = o3d.utility.Vector3dVector(camera_centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls


if __name__ == "__main__":
    example_dir = Path(
        "/home/ylao/repo/Open3D/examples/Python/ReconstructionSystem")
    fragment_dir = example_dir / "dataset/realsense/fragments"
    scene_dir = example_dir / "dataset/realsense/scene"

    all_frame_Ts = []
    for fragment_id in range(2):
        pose_graph_path = fragment_dir / f"fragment_optimized_{fragment_id:03d}.json"
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
        all_frame_Ts.append(Ts)

        # Get point cloud
        pcd_path = fragment_dir / f"fragment_{fragment_id:03d}.ply"
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        # Get poses as lineset
        camera_centers_ls = get_camera_centers_lineset(Ts)

        # Get reference coordinates
        camera_frames = get_camera_frames(Ts, size=0.02)

        # Visualize
        o3d.draw_geometries([pcd, camera_centers_ls, camera_frames])
        # o3d.draw_geometries([pcd])

    scene_mesh_path = scene_dir / "integrated.ply"
    scene_mesh = o3d.io.read_triangle_mesh(str(scene_mesh_path))

    scene_pose_graph_path = scene_dir / "refined_registration_optimized.json"
    scene_pose_graph = o3d.io.read_pose_graph(str(scene_pose_graph_path))
    num_fragments = len(scene_pose_graph.nodes)

    registered_Ts = []
    for fragment_id in range(num_fragments):
        pose = scene_pose_graph.nodes[fragment_id].pose
        for T in all_frame_Ts[fragment_id]:
            registered_T = T @ np.linalg.inv(pose)
            registered_Ts.append(registered_T)

    camera_frames = get_camera_frames(registered_Ts, size=0.02)
    camera_centers_ls = get_camera_centers_lineset(registered_Ts)
    o3d.draw_geometries([scene_mesh, camera_frames, camera_centers_ls])
    # o3d.draw_geometries([scene_mesh])
