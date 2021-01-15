#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=6G 
 
#SBATCH --job-name=biobank2_rgb_pairs
#SBATCH --output=output/biobank2_rgb_pairs.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4

cd ../.. && \
pip install --user -r requirements.txt && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank1 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 30 && \ 

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank3 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 30 && \

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank4 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 30 && \

python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 2 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 4 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 6 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 8 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 10 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 12 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 14 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 16 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 18 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 20 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 22 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 24 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 26 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 28 && \
python3 test.py --dataroot ./datasets/biobanks_pairs/biobank5 --name biobank2_rgb_pairs --model test --dataset_mode unaligned --load_size 1024 --crop_size 1024 --norm batch --epoch 30