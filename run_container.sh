#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
echo 'Looking for GPUs (ETA: 10 seconds)'
gpu=$(lspci | grep -i '.* vga .* nvidia .*')
shopt -s nocasematch
if [[ $gpu == *' nvidia '* ]]; then
  echo GPU found
  docker run --name astromer_v2 -it \
    --rm \
    --privileged \
    --mount "type=bind,src=$(pwd),dst=/home/" \
    --workdir /home/ \
    -p 8888:8888 \
    -p 6006:6006 \
    --gpus all \
    -e HOST="$(whoami)" \
    astromer bash
else
  docker run --name astromer_v2 -it \
    --rm \
    --privileged \
    --mount "type=bind,src=$(pwd),dst=/home/" \
    --workdir /home/ \
    -p 8888:8888 \
    -p 6006:6006 \
    -e HOST="$(whoami)" \
    astromer bash
fi
