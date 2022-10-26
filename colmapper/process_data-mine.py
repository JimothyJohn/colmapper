# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
# Borrowed heavily from https://github.com/nerfstudio-project/nerfstudio/blob/main/scripts/process_data.py

import subprocess
import sys
from enum import Enum
from typing import List, Optional
import tarfile
import os.path
import json
from pathlib import Path

from typing_extensions import Literal
import numpy as np
from nerfstudio.utils import colmap_utils


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        tar.close()


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


def get_colmap_version(default_version=3.7) -> float:
    """Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command("colmap", verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            return float(line.split(" ")[1])
    return default_version


def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        sys.exit(1)
    if out.stdout is not None:
        output = out.stdout.decode("utf-8")
        return output[:10]

    return out


def get_num_frames_in_video(video: Path) -> int:
    """Returns the number of frames in a video.

    Args:
        video: Path to a video.

    Returns:
        The number of frames in a video.
    """
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets \
            -show_entries stream=nb_read_packets -of csv=p=0 {video}"
    output = run_command(cmd)
    assert output is not None
    output = output.strip(" ,\t\n\r")
    return int(output)


def convert_video_to_images(
    video_path: Path,
    image_dir: Path,
    num_frames_target: int = 150,
    verbose: bool = False,
) -> List[str]:
    """Converts a video into a sequence of images.

    Args:
        video_path: Path to the video.
        output_dir: Path to the output directory.
        num_frames_target: Number of frames to extract.
        verbose: If True, logs the output of the command.
    Returns:
        A summary of the conversion.
    """

    # delete existing images in folder
    for img in image_dir.glob("*.jpg"):
        img.unlink()

    num_frames = get_num_frames_in_video(video_path)
    if num_frames == 0:
        sys.exit(1)

    out_filename = image_dir / "frame_%05d.jpg"
    ffmpeg_cmd = f"ffmpeg -i {video_path}"
    spacing = num_frames // num_frames_target

    if spacing > 1:
        ffmpeg_cmd += f" -vf 'thumbnail={spacing},setpts=N/TB' -r 1"

    # Reduce resolution to increase processing speed
    ffmpeg_cmd += f" -vf 'scale=640:-2' {out_filename}"

    run_command(ffmpeg_cmd, verbose=verbose)

    summary_log = []
    summary_log.append(f"Starting with {num_frames} video frames")
    summary_log.append(f"We extracted {len(list(image_dir.glob('*.jpg')))} images")

    return summary_log


def colmap_to_json(
    cameras_path: Path, images_path: Path, output_dir: Path, camera_model: CameraModel
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.
    Args:
        cameras_path: Path to the cameras.bin file.
        images_path: Path to the images.bin file.
        output_dir: Path to the output directory.
        camera_model: Camera model used.
    Returns:
        The number of registered images.
    """

    cameras = colmap_utils.read_cameras_binary(cameras_path)
    images = colmap_utils.read_images_binary(images_path)

    # Only supports one camera
    camera_params = cameras[1].params

    frames = []
    for _, im_data in images.items():
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = Path(f"./images/{im_data.name}")

        frame = {
            "file_path": str(name),
            "transform_matrix": c2w.tolist(),
        }
        frames.append(frame)

    out = {
        "fl_x": float(camera_params[0]),
        "fl_y": float(camera_params[1]),
        "cx": float(camera_params[2]),
        "cy": float(camera_params[3]),
        "w": cameras[1].width,
        "h": cameras[1].height,
        "camera_model": camera_model.value,
    }

    if camera_model == CameraModel.OPENCV:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "p1": float(camera_params[6]),
                "p2": float(camera_params[7]),
            }
        )
    if camera_model == CameraModel.OPENCV_FISHEYE:
        out.update(
            {
                "k1": float(camera_params[4]),
                "k2": float(camera_params[5]),
                "k3": float(camera_params[6]),
                "k4": float(camera_params[7]),
            }
        )

    out["frames"] = frames

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "sequential",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        verbose: If True, logs the output of the command.
        matching_method: "sequential",
    """

    # Feature extraction
    feature_extractor_cmd = [
        "colmap feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        "--ImageReader.camera_model OPENCV",
    ]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    print(f"Extracting features...")
    run_command(feature_extractor_cmd, verbose=False)

    # Feature matching
    feature_matcher_cmd = [
        f"colmap {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
    ]
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    print(f"Matching features...")
    run_command(feature_matcher_cmd, verbose=False)

    # Feature mapping
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        "colmap mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]

    """
    colmap_version = get_colmap_version()
    if colmap_version >= 3.7:
        mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")
    """

    mapper_cmd = " ".join(mapper_cmd)
    print(f"Mapping features")
    run_command(mapper_cmd, verbose=verbose)

    # Feature mapping
    bundle_dir = sparse_dir / "0"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_adjuster_cmd = [
        "colmap bundle_adjuster",
        f"--input_path {bundle_dir}",
        f"--output_path {bundle_dir}",
    ]
    print(f"Adjusting bundles")
    run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
