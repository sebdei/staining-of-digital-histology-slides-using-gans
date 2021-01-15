#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=6G 
 
#SBATCH --job-name=biobank1_lab_full
#SBATCH --output=output/biobank1_lab_full.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4

cd ../.. && \
pip install --user -r requirements.txt && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank2 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 30 && \ 

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 30 && \

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 30 && \

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank1_lab_full --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --no_dropout --model_suffix _A --color_space LAB --epoch 30