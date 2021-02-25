#!/bin/bash -l
#SBATCH -p datasci3,datasci4,datasci
#SBATCH --job-name=pgd_attack
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4




python pgd_attack_mx.py --epsilon 8 --num-steps 10 --n_classes 10 --attack_type pgd

python mlp_attack.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd --source $model_name

python bce_attack_mx.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd