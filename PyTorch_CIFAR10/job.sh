#!/bin/bash -l
#SBATCH -p datasci3,datasci4,datasci
#SBATCH --job-name=pgd_attack
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4



# multi-class pgd attack  8/255
python pgd_attack_mx.py --epsilon 8 --num-steps 10 --n_classes 10 --attack_type pgd

# binary class pgd attack on mlp 16/255
python mlp_attack.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd --source $model_name

# binary class pgd attack on CNN 16/255
python bce_attack_mx.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd

# gaussian noise attack of cnn on binary class
python evaluation_gaussian_noise.py --dataset cifar10 --n_classes 2 --epsilon 0.2 --seed 0 --cnn 1
python evaluation_gaussian_noise.py --dataset cifar10 --n_classes 2 --epsilon 0.5 --seed 0 --cnn 1
