#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=6G 
 
#SBATCH --job-name=pairs_hsv_2048_evaluation
#SBATCH --output=output/pairs_hsv_2048.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sebastian.deisel@uni-muenster.de

module load palma/2019b
module load fosscuda/2019b

module load PyTorch/1.6.0-Python-3.7.4
module load OpenCV/4.2.0-Python-3.7.4
module load scikit-learn/0.21.3-Python-3.7.4

cd ../../.. && \
pip install --user -r requirements.txt && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 2 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 4 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 6 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 8 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 10 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 12 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 14 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 16 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 18 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 20 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 22 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 24 && \
# python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 26 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 28 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 30 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 32 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 34 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 36 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 38 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 40 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 42 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 44 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 46 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 48 && \
python3 test.py --dataroot ./datasets/pairs --name pairs_hsv_2048 --model test --dataset_mode aligned --load_size 2048 --crop_size 2048 --norm batch --num_test 200 --color_space HSV --epoch 50