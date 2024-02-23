#!/usr/bin/env bash

# Some distro requires that the absolute path is given when invoking lspci
# e.g. /sbin/lspci if the user is not root.
gpu=$1
true="true"

echo GPU found
docker run  -it \
  --rm \
  --privileged=true \
  --mount "type=bind,src=$(pwd),dst=/home/" \
  --workdir /home/ \
  -p 8852:8852 \
  -p 1052:1052\
  --gpus all \
  -e HOST="$(whoami)" \
  astromer bash
