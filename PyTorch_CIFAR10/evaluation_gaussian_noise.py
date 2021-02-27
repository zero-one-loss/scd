import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import save_checkpoint, print_title, load_data
from tools.flag import args
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch





if __name__ == '__main__':

    np.random.seed(args.seed)

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)

    if args.cnn:
        train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))

    source_list = [
        'cifar10_binary_toy3rrr_nb2_bce_bp_8',
        'cifar10_binary_toy3srr100scale_nb2_bce_bp_8',
        'cifar10_binary_toy3ssr100scale_nb2_bce_bp_8',
        'cifar10_binary_toy3sss100scale_nb2_bce_bp_8',
        'cifar10_binary_toy3ssss100scale_nb2_bce_bp_8',
        'cifar10_binary_toy3srr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssr100scale_abp_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3sss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssss100scale_abp_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3srr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssr100scale_abp_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3sss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssss100scale_abp_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',

        'cifar10_binary_toy3rrr100_nb2_bce_adv_bp_8',
        'cifar10_binary_toy3srr100scale_nb2_bce_adv_bp_8',
        'cifar10_binary_toy3ssr100scale_nb2_bce_adv_bp_8',
        'cifar10_binary_toy3sss100scale_nb2_bce_adv_bp_8',
        'cifar10_binary_toy3ssss100scale_nb2_bce_adv_bp_8',
        'cifar10_binary_toy3srr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_bce_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3sss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_bce_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3srr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.1_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssr100scale_abp_adv_sign_i1_01loss_b200_lrc0.025_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3sss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.05_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        'cifar10_binary_toy3ssss100scale_abp_adv_sign_i1_01loss_b200_lrc0.05_lrf0.7_nb2_nw0_dm0_upc1_upf1_ucf32_normal_8',
        # f'{args.source}_{i}' for i in range(8)
    ]

    for j, source_model in enumerate(source_list):

        with open(f'../experiments/checkpoints/{source_model}.pkl', 'rb') as f:
            scd = pickle.load(f)
        print(f'Model: {source_model}',)
        yp = scd.predict(test_data).round()
        print('clean accuracy: ', accuracy_score(test_label, yp))
        acc = []
        for i in range(10):
            # print('%d run on epsilon %.3f:' % (i+1, args.epsilon))
        # if scd_args.normal_noise:
            noise = np.random.normal(0, 1, size=test_data.shape)
            noisy = np.clip((test_data + noise * args.epsilon), 0, 1)
            yp = scd.predict(noisy).round()
            temp_acc = accuracy_score(test_label, yp)
            # print('noise accuracy: ', temp_acc)
            acc.append(temp_acc)
        print('Average accuracy on epsilon %.3f: ' % args.epsilon, np.mean(acc))
