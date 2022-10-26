# Deploy Colmap

[![Replicate](https://replicate.com/jimothyjohn/colmap/badge)](https://replicate.com/jimothyjohn/colmap)

Are you as sick and tired as I am of trying to extract camera poses for your NeRF's? Well if so I've got a treat for you: A fully deployed, GPU-accelerated Colmap endpoint that can turn your video into a fully-configured Colmap workspace!

## Quickstart

Installs Docker and Cog, pulls image from Replicate, and runs on a [sample video](https://whatagan.s3.amazonaws.com/LionStatue.MOV) from my trip to the Louvre.

```bash
utils/Quickstart.sh
```

## Requirements

* CUDA>=11.6
* [Docker](https://www.docker.com)
* [Cog](https://github.com/replicate/cog#install)

## To-do

* Build tests
* Genericize approach

### Acknowledgements

- [COLMAP](https://github.com/colmap/colmap) for their amazing toolset that is used in EVERY NeRF deployment.
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio/blob/main/scripts) for their Python scripts.
- [svox2](https://github.com/sxyu/svox2) for their neural network-free radiance field utilities.
