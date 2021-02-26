#!/bin/bash -l
#SBATCH -p datasci3,datasci4,datasci
#SBATCH --job-name=cifar10_${seed}1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
cd ..
gpu=0


seed=0

#  traditional toy3 with back-propagation
python train_mlp.py --n_classes 2 --no_bias 0 --seed $seed --version mlprr --lr 0.001 --target cifar10_binary_mlprr_nb0_bce_bp_${seed}

# ABP training for bce
python train_cnn01_01.py --nrows 0.75 --localit 1  --updated_fc_features 128 --updated_fc_nodes 1  --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss bce --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc2 0.17 --version mlp01scale --seed $seed --iters 1000 --dataset cifar10 --n_classes 2 -- cnn 0 --divmean 0 --target cifar10_binary_mlp01scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb0_nw0_dm0_upc1_upf1_ucf32_normal_${seed} --updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 1 

# ABP training for 01loss
python train_cnn01_01.py --nrows 0.75 --localit 1  --updated_fc_features 128 --updated_fc_nodes 1  --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss 01loss --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc2 0.17 --version mlp01scale --seed $seed --iters 1000 --dataset cifar10 --n_classes 2 -- cnn 0 --divmean 0 --target cifar10_binary_mlp01scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb0_nw0_dm0_upc1_upf1_ucf32_normal_${seed} --updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 1


# if you want to do 8 votes

for seed in {0..7}
do
#  traditional toy3 with back-propagation
python train_mlp.py --n_classes 2 --no_bias 0 --seed $seed --version mlprr --lr 0.001 --target cifar10_binary_mlprr_nb0_bce_bp_${seed}

# ABP training for bce
python train_cnn01_01.py --nrows 0.75 --localit 1  --updated_fc_features 128 --updated_fc_nodes 1  --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss bce --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc2 0.17 --version mlp01scale --seed $seed --iters 1000 --dataset cifar10 --n_classes 2 -- cnn 0 --divmean 0 --target cifar10_binary_mlp01scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb0_nw0_dm0_upc1_upf1_ucf32_normal_${seed} --updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 1 

# ABP training for 01loss
python train_cnn01_01.py --nrows 0.75 --localit 1  --updated_fc_features 128 --updated_fc_nodes 1  --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss 01loss --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc2 0.17 --version mlp01scale --seed $seed --iters 1000 --dataset cifar10 --n_classes 2 -- cnn 0 --divmean 0 --target cifar10_binary_mlp01scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb0_nw0_dm0_upc1_upf1_ucf32_normal_${seed} --updated_fc_ratio 10 --updated_conv_ratio 20 --verbose_iter 1 

done



# combine votes

python combine_vote_mlp.py --dataset cifar10 --n_classes 2 --votes 8 --no_bias 0 --scale 1 -- cnn 0 --version toy3rrr100 --act sign --target cifar10_binary_toy3rrr100_nb0_bce_bp --save

python combine_vote_mlp.py --dataset cifar10 --n_classes 2 --votes 8 --no_bias 0 --scale 1 -- cnn 0 --version mlp01scale --act sign --target cifar10_binary_mlp01scale_nb0_bce_bp --save


