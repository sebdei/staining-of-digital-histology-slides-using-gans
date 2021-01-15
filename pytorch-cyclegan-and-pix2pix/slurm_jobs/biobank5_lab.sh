#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gputitanrtx
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=6G 
 
#SBATCH --job-name=biobank5_lab_full
#SBATCH --output=output/biobank5_lab_full.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4

cd .. && \
pip install --user -r requirements.txt && \
python3 train.py --dataroot ./datasets/biobanks_full/Biobank5 --name biobank5_lab_full --model cycle_gan --load_size 1024 --crop_size 768 --save_epoch_freq 2 --color_space LAB
