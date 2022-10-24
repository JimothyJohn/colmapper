#!/usr/bin/env bash

# Install utilities if needed
if [[ $(which docker) == "" ]]
then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh ./get-docker.sh
    rm ./get-docker.sh
fi

if [[ $(which cog) == "" ]]
then
    echo "Installing Cog..."
    sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
    sudo chmod +x /usr/local/bin/cog
fi

# Grab user inputs
WORKSPACE="workspace"
PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            echo "Optional args: -i or --input, -w or --workspace"
            exit 1
            ;;
        -i|--input)
            VIDEO="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -*|--*=) # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            shift
            ;;
        *) # preserve positional arguments
            PARAMS="$PARAMS $1"
            ;;
    esac
done

# Clear previous workspace
rm -r $WORKSPACE/*
mkdir -p $WORKSPACE/images

if [[ $VIDEO == "" ]]
then
    echo "Slicing lion sample"
    sudo cog run wget https://whatagan.s3.amazonaws.com/LionStatue.MOV && \
        ffmpeg -i LionStatue.MOV -r 3 -s 640x480 $WORKSPACE/images/%04d.jpg
fi

# Build reconstruction
sudo cog run colmap feature_extractor \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1

sudo cog run colmap sequential_matcher \
    --database_path $WORKSPACE/database.db \

mkdir -p $WORKSPACE/sparse/0
sudo cog run colmap mapper \
    --database_path $WORKSPACE/database.db \
    --image_path $WORKSPACE/images \
    --output_path $WORKSPACE/sparse

sudo cog run colmap bundle_adjuster \
    --input_path $WORKSPACE/sparse/0 \
    --output_path $WORKSPACE/sparse/0
