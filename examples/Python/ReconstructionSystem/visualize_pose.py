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


def flip_geometry(geometry):
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    geometry.transform(flip_transform)
    return geometry


if __name__ == "__main__":

    root_dir = Path("/home/yixing/data/redwood_recon/bedroom")
    mesh_path = root_dir / "ours_bedroom" / "bedroom.ply"
    pose_path = root_dir / "pose_bedroom" / "bedroom.log"

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    pose_graph = o3d.io.read_pinhole_camera_trajectory(str(pose_path))

    Ts = []
    camera_centers = []
    for camera_parameter in pose_graph.parameters[::15]:
        T = camera_parameter.extrinsic
        Ts.append(T)
        camera_center = plot3d.cameracenter_from_T(T)
        camera_centers.append(camera_center)

    # Get poses as lineset
    camera_centers_ls = get_camera_centers_lineset(Ts)

    # Get reference coordinates
    camera_frames = get_camera_frames(Ts, size=0.03)

    # Visualize
    o3d.visualization.draw_geometries([mesh, camera_centers_ls, camera_frames])
