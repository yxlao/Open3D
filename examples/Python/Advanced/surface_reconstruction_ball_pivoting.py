# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/surface_reconstruction_ball_pivoting.py

import open3d as o3d
import numpy as np
import os

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes

if __name__ == "__main__":
    gt_mesh = meshes.bunny()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(10000)
    o3d.io.write_point_cloud("bunny.ply", pcd)
    radii = [0.02, 0.05]

    o3d.visualization.draw_geometries([pcd])
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])

    mesh = rec_mesh
    target_number_of_triangles = arget_number_of_triangles = np.asarray(mesh.triangles).shape[0] // 2
    mesh_smp = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles)
    print("quadric decimated mesh has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.triangles).shape[0],
            np.asarray(mesh_smp.vertices).shape[0]))
    o3d.visualization.draw_geometries([mesh_smp])
