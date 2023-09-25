#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
gpu=$1
true="true"
echo "$gpu" 
if [[ $gpu == $true ]]; then
  echo GPU found
  docker run --name astromer_v2 -it \
    --rm \
    --privileged=true \
    --mount "type=bind,src=$(pwd),dst=/home/" \
    --workdir /home/ \
    -p 8885:8885 \
    -p 6007:6007 \
    --gpus all \
    -e HOST="$(whoami)" \
    astromer bash
else
  docker run --name astromer_v2 -it \
    --rm \
    --privileged=true \
    --mount "type=bind,src=$(pwd),dst=/home/" \
    --workdir /home/ \
    -p 8883:8883 \
    -p 6003:6003 \
    -e HOST="$(whoami)" \
    astromer bash
fi