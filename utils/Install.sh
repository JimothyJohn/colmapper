#!/usr/bin/env bash

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh

sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
