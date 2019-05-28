from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d

example_dir = Path(
    "/home/ylao/repo/Open3D/examples/Python/ReconstructionSystem")
fragment_dir = example_dir / "dataset/realsense/fragments"

pose_graph_path = fragment_dir / "fragment_optimized_000.json"
pose_graph = o3d.io.read_pose_graph(str(pose_graph_path))

Ts = []
for i in range(len(pose_graph.nodes)):
    pose = pose_graph.nodes[i].pose
    trans = np.linalg.inv(pose)
    Ts.append(trans)
print(len(Ts))

plot3d.plot_cameras(Ts)
