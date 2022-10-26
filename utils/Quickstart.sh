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
VIDEO="https://whatagan.s3.amazonaws.com/LionStatue.MOV"
PARAMS=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Flags
        -h|--help)
            echo "Optional args: -v or --video"
            exit 1
            ;;
        -v|--video)
            VIDEO="$2"
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

sudo cog predict -i $VIDEO
