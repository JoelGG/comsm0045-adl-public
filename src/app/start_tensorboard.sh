PORT=$((($UID-6025) % 65274))
hostname -s

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"
tensorboard --logdir "tensorboard_logs" --port "$PORT" --bind_all