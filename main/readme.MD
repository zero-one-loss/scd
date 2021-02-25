Training script
=

Guide
-

-   `train_cnn01_01.py` Script for binary classification.
-   `train_cnn01_abp_ce.py` Script for multi classes classification.

Flow path
-

-   Generate structure code
-   Train model
-   Combine votes


Examples
-

-   Binary classification

        python train_cnn01.py --nrows 0.75 --localit 1 --updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 --updated_conv_nodes 1 --width 100 --normalize 1 --percentile 1 --fail_count 1 --loss bce --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 --w-inc1 0.17 --w-inc2 0.2 --version fc --seed 0 --iters 1000 --dataset cifar10 --n_classes 2 --cnn 0 --divmean 0 --target cifar10_fc_sign_i1_bce_nb0_nw1_dm0_s1_fp32_32_0.pkl --updated_fc_ratio 1 --updated_conv_ratio 1


-   Multiple classes classification

        python train_cnn01_ce.py --nrows 0.03 --localit 1 --updated_conv_features 32 --updated_fc_features 128 --updated_fc_nodes 1 --updated_conv_nodes 1 --width 100 --normalize 0 --percentile 1 --fail_count 1 --loss mce --act sign --fc_diversity 1 --init normal --no_bias 1 --scale 2 --w-inc1 0.3 --w-inc2 0.2 --version toy2 --seed 0 --iters 20000 --dataset cifar10 --n_classes 10 --cnn 1 --divmean 0 --target cifar10_toy2_sign_i1_mce_nb1_nw0_dm0_s2_fp32_32_0.pkl --updated_fc_ratio 1 --updated_conv_ratio 1

-   Combine votes

    - Binary
    
            python combine_vote.py --dataset cifar10 --n_classes 2 --votes 32 --scale 1 --version fc --act sign --target cifar10_fc_sign_i1_bce_nb0_nw1_dm0_s1_fp32_32
        
    - Multiple classes
            
            Todo
    
   

Parameters for training script
-


-   `nrows` The data sampling ratio in each iteration. `0.75` means 75% data from **each class** will be randomly picked
in each iteration.

-   `localit` The maximum local iterations in scd search. Bigger iterations means deeper search.

-   `w-inc1` The step size for the convolutional layer in CNN structure, or the first layer in FC structure.

-   `w-inc2` The step size for the fc layer in CNN structure, or the second layer in FC structure.

-   `iters`  The number of iterations.

-   `updated_fc_features` The number of features/coordinates in fully connected layer will be considered to update in each iteration. 
`128` means scd will go over randomly 128 features' updating with the same step size
 and keep the best one among these 128 features.

-   `updated_conv_features` The number of features/coordinates in convolutional layer will be considered to update in each iteration. 
`32` means scd will go over randomly 32 features' updating with the same step size
 and keep the best one among these 32 features.
 
-   `interval` How many neighbors will be consider in local bias search.

-   `save` Whether to save the model.

-   `resume` Loading pretrained model based on `source`.

-   `n_classes` The number of classes, a parameter for loading data.

-   `target`  Save model named as `target`. For multiple runs, 
please set it as `$SharedName_#votes_#seed.pkl`(ex. `fc_100_0.pkl` for seed 0, `fc_100_1.pkl` for seed 1)
checkpoints will be saved in `checkpoints/pt` and named as `$SharedName_#votes_#seed.pt`. Log file will be 
saved in `logs/$dataset` such as `logs/cifar10/fc_100_0.csv`.

-   `dataset` Data set name, a parameter for loading data.

-   `version` network structure, 

    Please go to `core/cnn01.py` to see the details.

-   `seed`  Global random seed.

-   `fp16`  Enable fp16 computing.

-   `act`   Activation function in each layer except the last layer. (option: 'relu', 'sigmoid', 'sign')

-   `normalize`  Whether to normalize the weights.

-   `percentile`  Split bias in convolutional layer based on the percentile of all projection or the unique projection. 1 for all, 0 for unique

-   `sigmoid` Should be True if loss is binary cross entropy, will be set automatically.

-   `softmax` Should be True if loss is Cross-Entropy, will be set automatically.

-   `fail_count` Maximum fail time in deep search.
    
-   `width` A parameter to save GPU memory usage. If program give you a **"out of GPU memory"**
error, reduce it.    

-   `init` Distribution of initialized weights. `normal` means Normal distribution, `uniform` 
means uniform distribution. 

-   `no_bias` Whether contain bias in layer. `0` means each layer has bias, `1` means no bias except the last output layer, `2` means no bias for all layers.

-   `divmean`  Parameter for additional pre-processing for input data.

-   `verbose_iter` Frequency for showing and record training accuracy and test accuracy.

-   `cnn` After loading vector data, whether transpose it into image shape (Batch, Channel, Height, Width). `0` means no additional transpose, using data directly after loading.  

-   `freeze_layer` Freeze # layers during the training. 1 means freeze the first
layer, 2 means freeze the first two layers.

-   `temp_save_per_iter` During the training, save a temporary checkpoints every # iters

-   `lr_decay_iter` Decay learning rate every # iter.

-   `batch_increase_iter` Double the batch size for scd optimization every # iters.

-   `aug` Add it will do data augmentation for back-propagation part.

-   `balanced_sampling` Balanced sampling for scd part, default is true.

-   `bnn_layer` Default is False, Set it to 1 if you want to do binary weights training.

-   `epsilon` Epsilon for PGD attack during adversarial training.

-   `step` Step for PGD attack during adversarial training.

-   `alpha` alpha for PGD attack during adversarial training.

-   `bp_layer` ABP version, default is 0. 1 means the last layer is trained 
by back-propagation, 2 means the last two layers are trained by back-propagation, etc.

-   `lr` Learning rate for back-propagation.

-   `reinit` Re-initialize the layers' weights for the last # layers.

-   `bnn_layers` Ignore.

-   `adv_train` Adversarial training

-   `cuda` GPU enable

-   `loss` Loss function defined in `core/lossfunction.py`

-   `updated_fc_ratio` Increase it if GPU memory is not enough.

-   `updated_conv_ratio` Increase it if GPU memory is not enough.
    
