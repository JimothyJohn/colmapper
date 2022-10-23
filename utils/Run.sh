#!/usr/bin/env bash

# Grab user inputs
# Add arg formatting
VIDEO=$1
WORKSPACE=$2

# Create target directory
mkdir -p $WORKSPACE/images

# Slice video into frames
sudo cog run ffmpeg -i $VIDEO -r 3 $WORKSPACE/images/%04d.jpg

# Build reconstruction
sudo cog run colmap automatic_reconstructor \
    --workspace_path $WORKSPACE \
    --image_path $WORKSPACE/images
