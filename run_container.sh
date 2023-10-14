#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
gpu=$1
true="true"
echo "$gpu" 

echo GPU found
docker run  -it \
  --rm \
  --privileged=true \
  --mount "type=bind,src=$(pwd),dst=/home/" \
  --workdir /home/ \
  -p 8899:8899 \
  -p 6099:6099 \
  --gpus all 
  -e HOST="$(whoami)" \
  astromer bash

#if [[ $gpu == $true ]]; then
#  echo GPU found
#  docker run --name astromer_v2134 -it \
#    --rm \
#    --privileged=true \
#    --mount "type=bind,src=$(pwd),dst=/home/" \
#    --workdir /home/ \
#    -p 8886:8886 \
#    -p 6006:6006 \
#    --gpus all \
#    -e HOST="$(whoami)" \
#    astromer bash
#else
#  docker run --name astromer_v2134 -it \
#    --rm \
#    --privileged=true \
#    --mount "type=bind,src=$(pwd),dst=/home/" \
#    --workdir /home/ \
#    -p 8886:8886 \
#    -p 6006:6006 \
#    -e HOST="$(whoami)" \
#    astromer bash
#fi