#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
 
#SBATCH --job-name=biobank4_rgb_pairs
#SBATCH --output=output/biobank4_rgb_pairs.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4

cd .. && \
pip install --user -r requirements.txt && \
python3 train.py --dataroot ./datasets/biobanks_pairs_train/biobank4 --name biobank4_rgb_pairs --model pix2pix --load_size 1024 --crop_size 1024 --save_epoch_freq 2    --netG resnet_9blocks
