White box attack and transfer attack
=



Files
-


-   `mlp_attack.py` MLP (BCE loss and sigmoid output for binary classification)

        python mlp_attack.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd  --source $model_name

-   `bce_attack_mx.py` CNN models (BCE loss and sigmoid output for binary classification)

        python bce_attack_mx.py --epsilon 16 --num-steps 10 --n_classes 2 --attack_type pgd


-   `pgd_attack_mx.py` CNN models (CE loss and softmax output for multi-class classification)

        python pgd_attack_mx.py --epsilon 8 --num-steps 10 --n_classes 10 --attack_type pgd

-   `evaluation_gaussian_noise.py` Gaussian noise attack for binary classification models.
        
        python evaluation_gaussian_noise.py --dataset cifar10 --n_classes 2 --epsilon 0.2 --seed 0 --cnn 1
        
        python evaluation_gaussian_noise.py --dataset cifar10 --n_classes 2 --epsilon 0.5 --seed 0 --cnn 1

Notes
-

1. Copy the ensemble models define code from `scd/core/ensemble_model.py` 
to `core/ensemble_model.py` in the current directory. Remove `.round()` 
or `.argmax(dim=-1)` from the structure which you will use as source model to 
generate adversaries.

2. Change the checkpoints' path. The default path is from `scd/experiment/checkpoints`.

3. If your network has only one output node, please use  `mlp_attack.py` for flatten vectors,
`bce_attack_mx.py` for image (BCHW) vectors.
