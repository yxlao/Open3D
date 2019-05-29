from pathlib import Path
import open3d as o3d
import numpy as np
import plot3d
import cv2
import os

if __name__ == "__main__":
    example_dir = Path(
        "/home/ylao/repo/Open3D/examples/Python/ReconstructionSystem")
    dataset_dir = example_dir / "dataset" / "realsense"
    depth_dir = dataset_dir / "depth"
    depth_scaled_dir = dataset_dir / "depth_scaled"
    os.makedirs(depth_scaled_dir, exist_ok=True)

    image_id = 0
    for image_id in range(200):
        im_name = f"{image_id:06d}.png"
        im_path = depth_dir / im_name
        out_im_path = depth_scaled_dir / im_name

        im_depth = o3d.io.read_image(str(im_path))
        im_depth = np.asarray(im_depth)
        print("before", np.max(im_depth))

        im_depth = im_depth
        im_depth_scaled = cv2.applyColorMap(
            cv2.convertScaleAbs(im_depth, alpha=0.15), cv2.COLORMAP_JET)
        print("after", np.max(im_depth_scaled))

        cv2.imwrite(str(out_im_path), im_depth_scaled)
