import argparse
import os
from colmapper.utils import prepare_workdir, process_ply_file
from colmapper.video_processing import extract_frames
from colmapper.colmap import *

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", type=str, required=True, help="Path to the working directory")
parser.add_argument(
    "--datatype", type=str, default="llff", choices=["llff", "blender", "hypernerf"], help="Type of dataset"
)
parser.add_argument(
    "--downsample", type=int, default=1, help="Downsample factor for video frames"
)
args = parser.parse_args()

if __name__ == "__main__":
    prepare_workdir(args.workdir)
    extract_frames(args.workdir, args.downsample)
    run_colmap_script(args.workdir, args.datatype)
    process_ply_file(os.path.join(args.workdir, "colmap", "dense", "workspace", "fused.ply"), os.path.join(args.workdir, "points3D_downsample2.ply"))
