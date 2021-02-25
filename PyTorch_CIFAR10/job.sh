#!/bin/bash -l
#SBATCH -p datasci3,datasci4,datasci
#SBATCH --job-name=pgd_attack
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4


#python sgm_attack.py --source nob --target vgg19_bnn_normalize1 --epsilon 8 >> eps0.03125
#python sgm_attack.py --source nob --target vgg19_bnn_normalize1 --epsilon 16 >> eps0.0625
#python sgm_attack.py --source nob --target vgg19_bnn_normalize1 --epsilon 32 >> eps0.125
#python sgm_attack.py --source nob --target vgg19_bnn_normalize1 --epsilon 64 >> eps0.25
#
#python sgm_attack.py --source adv --target vgg19_bnn_normalize1 --epsilon 8 >> adv_eps0.03125
#python sgm_attack.py --source adv --target vgg19_bnn_normalize1 --epsilon 16 >> adv_eps0.0625
#python sgm_attack.py --source adv --target vgg19_bnn_normalize1 --epsilon 32 >> adv_eps0.125
#python sgm_attack.py --source adv --target vgg19_bnn_normalize1 --epsilon 64 >> adv_eps0.25
#


#python sgm_attack.py --source nob_normalize1 --target vgg19_bnn_normalize1 --epsilon 8 > nobn_eps0.03125
#python sgm_attack.py --source nob_normalize1 --target vgg19_bnn_normalize1 --epsilon 16 > nobn_eps0.0625
#python sgm_attack.py --source nob_normalize1 --target vgg19_bnn_normalize1 --epsilon 32 > nobn_eps0.125
#python sgm_attack.py --source nob_normalize1 --target vgg19_bnn_normalize1 --epsilon 64 > nobn_eps0.25


#python sgm_attack.py --source toy3 --target vgg19_bnn_normalize1 --gamma 2 --epsilon 8 > pgd_eps0.03125
#python sgm_attack.py --source toy3 --target vgg19_bnn_normalize1 --gamma 2 --epsilon 16 > pgd_eps0.0625
#python sgm_attack.py --source toy3 --target vgg19_bnn_normalize1 --gamma 2 --epsilon 32 > pgd_eps0.125
#python sgm_attack.py --source toy3 --target vgg19_bnn_normalize1 --gamma 2 --epsilon 64 > pgd_eps0.25


#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3srr100scale_abp_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm
#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3ssr100scale_abp_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm
#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3sss100scale_abp_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm

#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3srr100scale_abp_adv_thm_sign_i1_mce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm
#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3ssr100scale_abp_adv_thm_retrain0_sign_i1_mce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm
#python pgd_attack.py --gamma 2 --n_classes 10 --source cifar10_toy3sss100scale_abp_adv_thm_retrain0_sign_i1_mce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_fp16_8 --target vgg19_bn >> pgd_0208_thm

python pgd_attack_mx.py --gamma 2 --epsilon 8 --num-steps 10 --n_classes 10 > pgd_whitebox_0223_thm4_eps8