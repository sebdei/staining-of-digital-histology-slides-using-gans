#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gputitanrtx
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=6G 
 
#SBATCH --job-name=pairs_rgb_4096_evaluation
#SBATCH --output=output/pairs_rgb_4096.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4

cd ../../.. && \
pip install --user -r requirements.txt && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 2 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 4 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 6 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 8 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 10 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 12 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 14 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 16 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 18 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 20 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 22 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 24 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 26 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 28 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 30 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 32 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 34 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 36 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 38 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 40 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 42 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 44 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 46 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 48 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_rgb_4096 --model test --dataset_mode aligned --load_size 4096 --crop_size 4096 --norm batch --num_test 200 --epoch 50