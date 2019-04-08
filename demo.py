import open3d as o3d
import numpy as np

pcd_path = "/Users/yixing/repo/Open3D/examples/TestData/ICP/cloud_bin_0.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

vis = o3d.visualization.VisualizerWithEditing()
# vis = o3d.visualization.Visualizer()
vis.create_window("Visualizer")
vis.add_geometry(pcd)

step = 300
while (True):
    if len(np.asarray(pcd.points)) < step:
        break
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[:-step])
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[:-step])
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    print(len(np.asarray(pcd.points)))
