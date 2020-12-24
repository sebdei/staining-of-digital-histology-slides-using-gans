docker run --rm -it --init   --gpus=all   --ipc=host   --user="$(id -u):$(id -g)"   --volume="$PWD:/app"   anibali/pytorch bash
