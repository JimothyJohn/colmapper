#!/usr/bin/env bash

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
