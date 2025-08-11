# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
from shutil import make_archive
import shutil
import random
from cog import BasePredictor, Input, Path
from colmapper.nerfstudio.process_data import *
from colmapper.utils import prepare_workdir, process_ply_file, unzip_file
from colmapper.video_processing import extract_frames
from colmapper.colmap import *

quality_num = {"Low": 4, "Med": 2, "High": 1}

def process_dynamic(videos, workdir):
    
    unzip_file(videos, workdir)
    prepare_workdir(workdir)
    extract_frames(workdir, 1)
    run_colmap_script(workdir, "llff")
    process_ply_file(os.path.join(workdir, "colmap", "dense", "workspace", "fused.ply"), os.path.join(workdir, "points3D_downsample2.ply"))

    return Path(workdir)

class Predictor(BasePredictor):
    def predict(
        self,
        videos: Path = Input(description="Short sample video or zip file of multiple videos"),
        name: str = Input(description="Name of experiment", default="colmap-out"),
        scene_type: str = Input(
            description="Static or dynamic scene",
            choices=["static", "dynamic"],
            default="static",
        ),
        format: str = Input(
            description="Colmap output format ex: instant-ngp, nerfacto, arf",
            choices=["dynerf", "instant-ngp", "nerfacto", "arf"],
            default="nerfacto",
        ),
        quality: str = Input(
            description="Resolution of images",
            choices=["Low", "Med", "High"],
            default="Low",
        ),
    ) -> Path:
        """Create COLMAP zipfile from video"""
        filename = f"{name}"
        colmap_dir = Path(filename)
       
        # Start with catch-all matching but offer the faster, sequential option
        matching_method = "sequential"
        if scene_type == "dynamic":
            matching_method = "exhaustive"

        num_frames_target = 300 / (quality_num[quality])
        colmapper = ProcessVideo(
            data=video,
            output_dir=colmap_dir,
            num_frames_target=num_frames_target,
            matching_method=matching_method,
            num_downscales=quality_num[quality],
            verbose=True,
        )

        colmapper.main()

        if scene_type == "dynamic":
            workdir = "/tmp"
            process_dynamic(videos, workdir)
            make_archive(workdir, "zip", workdir)
            return Path(f"{workdir}.zip")
        else:
            make_archive(filename, "zip", filename)
            return Path(f"{filename}.zip")
