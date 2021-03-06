#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gputitanrtx
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G 
 
#SBATCH --job-name=val_staining_of_histology_slides
#SBATCH --output=output.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de
 
module load palma/2019b
module load fosscuda/2019b
module load PyTorch/1.6.0-Python-3.7.4

python3 /home/s/s_deis02/virtual-staining-with-gans/gan/cyclegan/predict_whole_slide.py --model_path saved_models/staining/G_AB_80.pth
