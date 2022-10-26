# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import tarfile
import shutil
import random
from turtle import down

from cog import BasePredictor, Input, Path

from colmapper.nerfstudio.process_data import *
from colmapper.svox2.colmap2nsvf import read_colmap_sparse
from colmapper.svox2.create_split import list_filter_dirs


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        tar.close()


def get_nvsf(
    sparse_dir: Path,
    scale: float = 1.0,
    gl_cam: bool = False,
    overwrite: bool = False,
    overwrite_no_del: bool = False,
    colmap_suffix: bool = False,
):
    sparse_dir = str(sparse_dir)
    if sparse_dir.endswith("/"):
        sparse_dir = sparse_dir[:-1]
    base_dir = os.path.dirname(os.path.dirname(sparse_dir))
    pose_dir = os.path.join(base_dir, "pose_colmap" if colmap_suffix else "pose")
    feat_dir = os.path.join(base_dir, "feature")
    base_scale_file = os.path.join(base_dir, "base_scale.txt")
    if os.path.exists(base_scale_file):
        with open(base_scale_file, "r") as f:
            base_scale = float(f.read())
        print("base_scale", base_scale)
    else:
        base_scale = 1.0
        print("base_scale defaulted to", base_scale)
    print("BASE_DIR", base_dir)
    print("POSE_DIR", pose_dir)
    print("FEATURE_DIR", feat_dir)
    print("COLMAP_OUT_DIR", sparse_dir)
    overwrite = overwrite

    def create_or_recreate_dir(dirname):
        if os.path.isdir(dirname):
            nonlocal overwrite
            if overwrite:
                if not overwrite_no_del:
                    shutil.rmtree(dirname)
                overwrite = True
            else:
                print("Quitting")
                sys.exit(1)

        os.makedirs(dirname, exist_ok=True)

    cameras, imdata, points3D = read_colmap_sparse(sparse_dir)
    create_or_recreate_dir(pose_dir)
    create_or_recreate_dir(feat_dir)

    print("Get intrinsics")
    K = np.eye(4)
    K[0, 0] = cameras[0].params[0] / base_scale
    K[1, 1] = cameras[0].params[0] / base_scale
    K[0, 2] = cameras[0].params[1] / base_scale
    K[1, 2] = cameras[0].params[2] / base_scale
    print("f", K[0, 0], "c", K[0:2, 2])
    np.savetxt(
        os.path.join(
            base_dir,
            "intrinsics_colmap.txt" if colmap_suffix else "intrinsics.txt",
        ),
        K,
    )
    del K

    print("Get world scaling")
    points = np.stack([p.xyz for p in points3D])
    cen = np.median(points, axis=0)
    points -= cen
    dists = (points**2).sum(axis=1)

    # FIXME: Questionable autoscaling. Adopt method from Noah Snavely
    meddist = np.median(dists)
    points *= 2 * scale / meddist

    # Save the sparse point cloud
    np.save(os.path.join(base_dir, "points.npy"), points)
    print(cen, meddist)

    print("Get cameras")

    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    coord_trans = np.diag([1, -1, -1, 1.0])
    for im in imdata:
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        xys = im.xys
        point3d_ids = im.point3D_ids
        #  w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        t_world = -R.T @ t
        t_world = (t_world - cen[:, None]) * 2 * scale / meddist
        c2w = np.concatenate([np.concatenate([R.T, t_world], 1), bottom], 0)

        if gl_cam:
            # Use the alternate camera space convention of jaxNeRF, OpenGL etc
            # We use OpenCV convention
            c2w = c2w @ coord_trans

        imfile_name = os.path.splitext(os.path.basename(im.name))[0]
        pose_path = os.path.join(pose_dir, imfile_name + ".txt")
        feat_path = os.path.join(
            feat_dir, imfile_name + ".npz"
        )  # NOT USED but maybe nice?
        np.savetxt(pose_path, c2w)
        np.savez(feat_path, xys=xys, ids=point3d_ids)
    print(" Total cameras:", len(imdata))
    print("Done!")


def get_split(root_dir: Path, every: int = 16, randomize: bool = False):
    root_dir = str(root_dir)
    dirs, dir_idx = list_filter_dirs(root_dir)

    refdir = dirs[dir_idx]
    print("going to split", [x.name for x in dirs], "reference", refdir.name)
    base_files = [
        os.path.splitext(x)[0]
        for x in sorted(os.listdir(refdir.name))
        if os.path.splitext(x)[1].lower() in refdir.valid_exts
    ]
    if randomize:
        print("random enabled")
        random.shuffle(base_files)
    base_files_map = {
        x: f"{int(i % every == 0)}_" + x for i, x in enumerate(base_files)
    }

    for dir_obj in dirs:
        dirname = dir_obj.name
        files = sorted(os.listdir(dirname))
        for filename in files:
            full_filename = os.path.join(dirname, filename)
            if filename.startswith("0_") or filename.startswith("1_"):
                continue
            if not os.path.isfile(full_filename):
                continue
            base_file, ext = os.path.splitext(filename)
            if ext.lower() not in dir_obj.valid_exts:
                print("SKIP ", full_filename, " Since it has an unsupported extension")
                continue
            if base_file not in base_files_map:
                print(
                    "SKIP ",
                    full_filename,
                    " Since it does not match any reference file",
                )
                continue
            new_base_file = base_files_map[base_file]
            new_full_filename = os.path.join(dirname, new_base_file + ext)
            print("rename", full_filename, "to", new_full_filename)
            os.rename(full_filename, new_full_filename)


quality_num = {"Low": 2, "Med": 1}


class Predictor(BasePredictor):
    def predict(
        self,
        video: Path = Input(description="Short sample video"),
        format: str = Input(
            description="Colmap output format ex: instant-ngp, nerfacto, arf",
            choices=["instant-ngp", "nerfacto", "arf"],
            default="instant-ngp",
        ),
        quality: str = Input(
            description="Resolution of images",
            choices=["Low", "Med", "High"],
            default="Low",
        ),
    ) -> Path:
        print(f"Formatting for {format}")
        """Create COLMAP archive from video"""
        colmap_dir = Path("cog-workspace")
        image_dir = colmap_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        convert_video_to_images(
            video_path=video,
            image_dir=image_dir,
            num_frames_target=150,
        )

        if quality != "High":
            downscale_images(image_dir=image_dir, num_downscales=quality_num[quality])

        run_colmap(
            image_dir=image_dir,
            colmap_dir=colmap_dir,
            camera_model=CameraModel.OPENCV,
            matching_method="sequential",
        )

        if format == "arf":
            sparse_dir = colmap_dir / "sparse" / "0"
            get_nvsf(sparse_dir, overwrite=True)
            get_split(colmap_dir, randomize=True)
        else:
            if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
                colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=colmap_dir,
                    camera_model=CameraModel.OPENCV,
                )

        make_tarfile("colmap.tar", "cog-workspace")
        return Path("colmap.tar")
