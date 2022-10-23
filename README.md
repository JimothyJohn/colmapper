# Deploy Colmap

Are you as sick and tired as I am of trying to extract camera poses for your NeRF's? Well if so I've got a treat for you: A fully deployed (but slow) Colmap endpoint that can turn your video into a fully-configured Colmap workspace!

## Quickstart

Installs Docker and Cog then pulls image from Replicate.

```bash
utils/Install.sh
```

## Requirements

* CUDA>=11.6
* Docker
* Cog

## Usage

Preface your colmap commands with "sudo cog run" like so:

```bash
sudo cog run colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images
```
