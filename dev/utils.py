import numpy as np
import os
import open3d as o3d

# https://github.com/hustvl/4DGaussians/blob/master/scripts/llff2colmap.py
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def prepare_workdir(workdir: str):
    """
    Prepares the workdir for colmap processing.
    """
    if not os.path.exists(workdir):
        raise ValueError(f"Invalid workdir: {workdir}")

    if not os.path.exists(os.path.join(workdir, "image_colmap")):
        os.makedirs(os.path.join(workdir, "image_colmap"))

    if not os.path.exists(os.path.join(workdir, "sparse_")):
        os.makedirs(os.path.join(workdir, "sparse_"))

    if not os.path.exists(os.path.join(workdir, "colmap")):
        os.makedirs(os.path.join(workdir, "colmap", "sparse", "0"))

    if not os.path.exists(os.path.join(workdir, "colmap", "dense", "workspace")):
        os.makedirs(os.path.join(workdir, "colmap", "dense", "workspace"))


# https://github.com/hustvl/4DGaussians/blob/master/scripts/downsample_point.py
def process_ply_file(input_file, output_file):
    # 读取输入的ply文件
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Total points: {len(pcd.points)}")

    # 通过点云下采样将输入的点云减少
    voxel_size=0.02
    while len(pcd.points) > 40000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)}")
        voxel_size+=0.01

    # 将结果保存到输入的路径中
    o3d.io.write_point_cloud(output_file, pcd)

