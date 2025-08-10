import os
from pathlib import Path
import numpy as np
import glob
import shutil
from utils import rotmat2qvec
import subprocess
from database import COLMAPDatabase, blob_to_array

# https://github.com/hustvl/4DGaussians/blob/master/scripts/llff2colmap.py
def llff2colmap(workdir: str) -> None:
    colmap_dir = os.path.join(workdir, "sparse_")

    poses_arr = np.load(os.path.join(workdir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    videos = glob.glob(os.path.join(workdir, "cam[0-9][0-9]"))
    videos = sorted(videos)
    assert len(videos) == poses_arr.shape[0]
    H, W, focal = poses[0, :, -1]
    focal = focal / 2
    focal = [focal, focal]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    videos = glob.glob(os.path.join(workdir, "cam[0-9][0-9]"))
    videos = sorted(videos)
    image_paths = []
    for index, video_path in enumerate(videos):
        image_path = os.path.join(video_path, "images", "0000.png")
        image_paths.append(image_path)
    goal_dir = os.path.join(workdir, "image_colmap")

    image_name_list = []
    for index, image in enumerate(image_paths):
        image_name = image.split("/")[-1].split(".")
        image_name[0] = "r_%03d" % index
        # breakpoint()
        image_name = ".".join(image_name)
        image_name_list.append(image_name)
        goal_path = os.path.join(goal_dir, image_name)
        shutil.copy(image, goal_path)

    # write image information.
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        for idx, pose in enumerate(poses):
            # pose_44 = np.eye(4)

            R = pose[:3, :3]
            R = -R
            R[:, 0] = -R[:, 0]
            T = pose[:3, 3]
            R = np.linalg.inv(R)
            T = -np.matmul(R, T)
            T_str = [str(i) for i in T]
            qevc = [str(i) for i in rotmat2qvec(R)]
            f.write(
                f"{idx+1} {' '.join(qevc)} {' '.join(T_str)} 1 {image_name_list[idx]}\n\n"
            )

    # write camera infomation.
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write(f"1 SIMPLE_PINHOLE 1352 1014 {focal[0]} {1352 / 2} {1014 / 2}\n")  #

    # Create an empty points3D.txt file, as it is expected by COLMAP.
    Path(os.path.join(colmap_dir, "points3D.txt")).touch()


# https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/colmap.sh#L15
def feature_extraction(workdir: str):
    """
    Uses subprocess to run this CLI function: colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
    --ImageReader.camera_model SIMPLE_PINHOLE
    --ImageReader.single_camera 1
    --SiftExtraction.gpu_index 0
    --SiftExtraction.use_gpu 1
    """
    subprocess.run(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            os.path.join(workdir, "colmap", "database.db"),
            "--image_path",
            os.path.join(workdir, "image_colmap"),
            "--SiftExtraction.max_image_size",
            "4096",
            "--SiftExtraction.max_num_features",
            "16384",
            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
            "--ImageReader.camera_model",
            "SIMPLE_PINHOLE",
            "--ImageReader.single_camera",
            "1",
        ]
    )

# https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/database.py#L52
# TODO explote implementing copy feature to move saprse_ to sparse_custom
def camTodatabase(workdir: str):
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}

    database_path = os.path.join(workdir, "colmap", "database.db")
    txt_path = os.path.join(workdir, "sparse_", "cameras.txt")
    # breakpoint()
    if os.path.exists(database_path)==False:
        print("ERROR: database path doesn't exist -- please run feature extraction first.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(txt_path, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    # Commit the data to the file.
    db.commit()
    # Read and check cameras.
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    # Close database.db.
    db.close()


# https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/colmap.sh#L15
# TODO check if function has already been run before running
def exhaustive_matcher(workdir: str):
    """
    Uses subprocess to run this CLI function: colmap exhaustive_matcher --database_path $workdir/colmap/database.db
    """
    subprocess.run(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            os.path.join(workdir, "colmap", "database.db"),
        ]
    )


# https://github.com/hustvl/4DGaussians/blob/843d5ac636c37e4b611242287754f3d4ed150144/colmap.sh#L20
def point_triangulator(workdir: str):
    """
    colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1
    """
    subprocess.run(
        [
            "colmap",
            "point_triangulator",
            "--database_path",
            os.path.join(workdir, "colmap", "database.db"),
            "--image_path",
            os.path.join(workdir, "image_colmap"),
            "--input_path",
            os.path.join(workdir, "sparse_"),
            "--output_path",
            os.path.join(workdir, "colmap","sparse", "0"),
            "--clear_points",
            "1",
        ]
    )



def image_undistorter(workdir: str):
    """
    colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/workspace
    """
    subprocess.run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            os.path.join(workdir, "image_colmap"),
            "--input_path",
            os.path.join(workdir, "colmap", "sparse", "0"),
            "--output_path",
            os.path.join(workdir, "colmap", "dense", "workspace"),
        ]
    )

def patch_match_stereo(workdir: str):
    """
    colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
    """
    subprocess.run(
        [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            os.path.join(workdir, "colmap", "dense", "workspace"),
        ]
    )


def stereo_fusion(workdir: str):
    """
    colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply
    """
    subprocess.run(
        [
            "colmap",
            "stereo_fusion",
            "--workspace_path",
            os.path.join(workdir, "colmap", "dense", "workspace"),
            "--output_path",
            os.path.join(workdir, "colmap", "dense", "workspace", "fused.ply"),
        ]
    )


# https://github.com/hustvl/4DGaussians/blob/master/colmap.sh
def run_colmap_script(workdir: str, datatype: str = "llff"):
    """
    Runs colmap shell CLI functions in a CLI wrapper in various stages.
    """
    if datatype == "llff":
        llff2colmap(workdir)
    else:
        raise ValueError(f"Invalid datatype: {datatype}")

    feature_extraction(workdir)
    camTodatabase(workdir)
    exhaustive_matcher(workdir)
    point_triangulator(workdir)
    image_undistorter(workdir)
    patch_match_stereo(workdir)
    stereo_fusion(workdir)

def generate_poses_for_scene(workdir: str, match_type: str = "exhaustive_matcher"):
    """
    Generates 'poses_bounds.npy' for a scene using 'gen_poses' from 'imgs2poses.py'.
    The scene is expected to be in 'workdir/first_frame/'.
    Skips if 'workdir/first_frame/poses_bounds.npy' already exists or if input images are missing.

    Args:
        workdir (str): The base working directory (e.g., 'cook_spinach').
                       The function will operate on the 'workdir/first_frame/' subdirectory.
        match_type (str): Type of matcher to use ('exhaustive_matcher' or 'sequential_matcher').
                          Defaults to 'exhaustive_matcher'.
    """
    scene_dir_for_poses = os.path.join(workdir, "first_frame")
    input_images_dir = os.path.join(scene_dir_for_poses, "images")
    expected_output_file = os.path.join(scene_dir_for_poses, "poses_bounds.npy")

    if not (
        os.path.isdir(input_images_dir)
        and glob.glob(os.path.join(input_images_dir, "*.png"))
    ):
        print(
            f"🤷 Input images (PNGs) not found in '{input_images_dir}'. Skipping pose generation."
        )
        return

    if os.path.exists(expected_output_file):
        print(
            f"✅ Pose file '{expected_output_file}' already exists. Skipping pose generation step."
        )
        return

    try:
        script_gen_poses(scenedir=scene_dir_for_poses, match_type=match_type)
    except ImportError:
        print(
            f"AI-generated: Error: Could not execute pose generation because 'gen_poses' from 'replicate.imgs2poses' was not imported successfully."
        )
        raise
    except Exception as e:
        print(
            f"AI-generated: An error occurred during pose generation for '{scene_dir_for_poses}': {e}"
        )
        raise
