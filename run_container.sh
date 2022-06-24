ocker run -it --rm \
  --mount "type=bind,src=$(pwd),dst=/home/" \
  --workdir /home/ \
  -p 8888:8888 \
  -p 6006:6006 \
  --gpus all \
  astromer bash
