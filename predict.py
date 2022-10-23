# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
# Borrowed heavily from https://github.com/nerfstudio-project/nerfstudio/blob/main/scripts/process_data.py

import subprocess
import sys
from enum import Enum
from typing import List, Optional

from typing_extensions import Literal
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Don't have a model...yet
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        video: Path = Input(description="Short sample video"),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)


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
        return out.stdout.decode("utf-8")
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
    video_path: Path, image_dir: Path, num_frames_target: int, verbose: bool = False
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
    for img in image_dir.glob("*.png"):
        img.unlink()

    num_frames = get_num_frames_in_video(video_path)
    if num_frames == 0:
        sys.exit(1)

    print("Number of frames in video:", num_frames)

    out_filename = image_dir / "frame_%05d.png"
    ffmpeg_cmd = f"ffmpeg -i {video_path}"
    spacing = num_frames // num_frames_target

    if spacing > 1:
        ffmpeg_cmd += f" -vf 'thumbnail={spacing},setpts=N/TB' -r 1"

    ffmpeg_cmd += f" {out_filename}"

    run_command(ffmpeg_cmd, verbose=verbose)

    summary_log = []
    summary_log.append(f"Starting with {num_frames} video frames")
    summary_log.append(f"We extracted {len(list(image_dir.glob('*.png')))} images")

    return summary_log


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "sequential",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
    """

    colmap_version = get_colmap_version()

    (colmap_dir / "database.db").unlink(missing_ok=True)

    # Feature extraction
    feature_extractor_cmd = [
        "colmap feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
    ]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    run_command(feature_extractor_cmd, verbose=verbose)

    # Feature matching
    feature_matcher_cmd = [
        f"colmap {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
    ]
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    run_command(feature_matcher_cmd, verbose=verbose)

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        "colmap mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    mapper_cmd = " ".join(mapper_cmd)

    run_command(mapper_cmd, verbose=verbose)
    bundle_adjuster_cmd = [
        "colmap bundle_adjuster",
        f"--input_path {sparse_dir}/0",
        f"--output_path {sparse_dir}/0",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
