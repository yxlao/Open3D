# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
import os
import urllib.request
import gzip
import tarfile
import shutil
import time
import open3d as o3d
from pathlib import Path


def bunny_mesh():
    bunny_path = '../../TestData/Bunny.ply'
    if not os.path.exists(bunny_path):
        print('downloading bunny mesh')
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        urllib.request.urlretrieve(url, bunny_path + '.tar.gz')
        print('extract bunny mesh')
        with tarfile.open(bunny_path + '.tar.gz') as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(os.path.dirname(bunny_path), 'bunny', 'reconstruction',
                         'bun_zipper.ply'), bunny_path)
        os.remove(bunny_path + '.tar.gz')
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), 'bunny'))
    return o3d.io.read_triangle_mesh(bunny_path)


if __name__ == "__main__":
    mesh = bunny_mesh()
    mesh.compute_vertex_normals()

    new_num_triangles = int(len(mesh.triangles) / 10)
    mesh_simple = o3d.geometry.simplify_quadric_decimation(mesh,
                                                           new_num_triangles)
    # o3d.draw_geometries([mesh_simple])
    mesh_simple = o3d.geometry.simplify_vertex_clustering(mesh,
                                                          0.008)
    mesh_simple.compute_vertex_normals()
    # o3d.draw_geometries([mesh_simple])

    pcd = o3d.geometry.PointCloud()
    mesh_simple = o3d.geometry.simplify_vertex_clustering(mesh,
                                                          0.005)
    pcd.points = mesh_simple.vertices
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pcd.points))
    # o3d.draw_geometries([pcd])

    octree = o3d.geometry.Octree(5)
    octree.convert_from_point_cloud(pcd)
    # o3d.draw_geometries([octree])

    data_dir = Path("../../TestData/")
    fragment_path = str(data_dir / "fragment.ply")
    pcd = o3d.io.read_point_cloud(fragment_path)
    # o3d.visualization.draw_geometries([pcd])

    octree = o3d.geometry.Octree(6)
    octree.convert_from_point_cloud(pcd)
    # o3d.draw_geometries([octree])

    voxel_grid = octree.to_voxel_grid()
    o3d.draw_geometries([voxel_grid])
