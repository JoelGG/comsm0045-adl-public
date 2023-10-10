#!/usr/bin/env bash

#SBATCH --job-name=CW
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./sbatch_logs/log_%j.out # STDOUT out
#SBATCH -e ./sbatch_logs/log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=2:40:00
#SBATCH --mem=8GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

python train.py \
  --learning-rate 5e-5 \
  --batch-size 128 \
  --worker-count 1 \
  --net shallow

python train.py \
  --learning-rate 5e-5 \
  --batch-size 128 \
  --worker-count 1 \
  --net deep

python train.py \
  --learning-rate 5e-5 \
  --batch-size 128 \
  --worker-count 1 \
  --net bbnn
