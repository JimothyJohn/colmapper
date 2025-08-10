# 4D Utilities

The goal of this repository is twofold:

1. Neatly organize multiple, static view video files of a single, dynamic scene into standard dataset structures (DyNeRF, Google Immersive, etc) to allow for easier research and development into novel training approaches of dynamic volumetric scenes.

2. Create an end-to-end training and rendering pipeline for novel view creation on cloud GPU's

## Features

[x] [Portable development environment](Dockerfile.cuda)

[x] [Multi-threaded frame extraction utility for DyNeRF](replicate/video_processing.py)

[x] [Colmap pipeline for DyNeRF](replicate/colmap.py)

[x] [CLI](replicate/cli.py)

[ ] [poses_bounds.npy utility for custom dataset](https://github.com/Wen-Hui-Ma/Generate-poses_bounds.npy)

[ ] Cloud deployment

[ ] [Novel view trajectory template](https://github.com/fyusion/llff?tab=readme-ov-file#generate-poses-for-new-view-path)

### TODO

[ ] Extend frame extraction to include Google Immersive and others

[ ] Modify data loader in training pipeline to be zero-indexed

## Resources

- [DyNeRF datasets](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)

- [Pretrained models](https://github.com/NVlabs/queen/releases/tag/v1.0-neurips24)
