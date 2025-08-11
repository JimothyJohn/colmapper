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
    voxel_size = 0.02
    while len(pcd.points) > 40000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)}")
        voxel_size += 0.01

    # 将结果保存到输入的路径中
    o3d.io.write_point_cloud(output_file, pcd)


# AI-generated comment: The following function extracts a ZIP archive to a specified directory.
# It includes error handling and ensures the destination directory exists.
def unzip_file(zip_path, dest_path):
    """
    Unzips a file to a destination folder.

    :param zip_path: The path to the .zip file.
    :param dest_path: The path to the destination folder.
    """
    # AI-generated comment: Check if the provided path to the zip file exists.
    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at '{zip_path}'")
        sys.exit(1)

    # AI-generated comment: Create the destination directory if it does not already exist.
    # This prevents errors during the extraction process.
    os.makedirs(dest_path, exist_ok=True)

    try:
        # AI-generated comment: Open and extract the contents of the zip file.
        # The 'with' statement ensures the file is properly closed even if errors occur.
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print(f"Successfully extracted '{zip_path}' to '{dest_path}'")
    except zipfile.BadZipFile:
        # AI-generated comment: Handle cases where the file is not a valid ZIP archive.
        print(f"Error: '{zip_path}' is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        # AI-generated comment: Catch any other exceptions that may occur during extraction.
        print(f"An error occurred: {e}")
        sys.exit(1)
