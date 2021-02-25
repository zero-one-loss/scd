import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from core.text_wrap import TextWrapper
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title, TextDataset
from core.lossfunction import ZeroOneLoss, BCELoss, CrossEntropyLoss
from core.cnn01 import *
from core.basic_module import *
from core.basic_function import evaluation_text, get_features
import pickle
import torchvision
import torchvision.transforms as transforms
import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torchtext.data import Iterator, BucketIterator
import torch
import os


def train_single_cnn01(scd_args, device=None, seed=None, data_set=None, fined_train_label=None,
                       target_class=None):

    start_time_ = time.time()
    if device is not None:
        scd_args.gpu = device
    if seed is not None:
        scd_args.seed = seed
    resume = False
    use_cuda = scd_args.cuda
    dtype = torch.float16 if scd_args.fp16 else torch.float32

    best_acc = 0

    # seed = 2047775

    print('Random seed: ', scd_args.seed)
    np.random.seed(scd_args.seed)
    torch.manual_seed(scd_args.seed)
    df = pd.DataFrame(columns=['epoch', 'train acc', 'test acc'])
    log_path = os.path.join('logs', scd_args.dataset)
    log_file_name = os.path.join(log_path, scd_args.target)

    if scd_args.save:
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    tokenize = lambda x: x.split()

    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenize, lower=True,
                      fix_length=None, batch_first=True)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    


    all = TextDataset(data_set[0], TEXT, LABEL, False, False)
    train = TextDataset(data_set[1], TEXT, LABEL, False, False)
    test = TextDataset(data_set[2], TEXT, LABEL, False, False)

    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=scd_args.embedding_path)
    TEXT.build_vocab(all, vectors=vectors)
    # LABEL.build_vocab(train)
    # train_loader, val_loader = BucketIterator.splits(
    #     datasets=(train, train),
    #     batch_sizes=(scd_args.nrows, scd_args.nrows),
    #     device=torch.device('cpu'),
    #     sort_key=lambda x: len(x.text),
    #     sort_within_batch=False,
    #     repeat=False,
    #
    # )

    train_loader = Iterator(train, batch_size=scd_args.nrows,
                           device=torch.device('cpu'), sort=False,
                         sort_within_batch=False, repeat=False)

    val_loader = Iterator(train, batch_size=1,
                           device=torch.device('cpu'), sort=False,
                         sort_within_batch=False, repeat=False)

    test_loader = Iterator(test, batch_size=1,
                           device=torch.device('cpu'), sort=False,
                         sort_within_batch=False, repeat=False, shuffle=False)

    net = scd_args.structure(
        num_classes=scd_args.num_classes, act=scd_args.act, sigmoid=scd_args.sigmoid,
        softmax=scd_args.softmax, ndim=200, drop_p=scd_args.drop_p)
    embedding = nn.Embedding(len(TEXT.vocab.vectors), 200)
    embedding.weight.data.copy_(TEXT.vocab.vectors)

    weight = embedding.weight
    weight.requires_grad = False
    norms = weight.data.norm(2, 1)
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    weight.data.div_(norms.expand_as(weight.data))
    weight[weight.ne(weight)] = 0
    if not scd_args.sigmoid and not scd_args.softmax:
        _01_init(net)
    else:
        init_weights(net, kind=scd_args.init)
    best_model = scd_args.structure(
        num_classes=scd_args.num_classes, act=scd_args.act, sigmoid=scd_args.sigmoid,
    softmax=scd_args.softmax, ndim=200, drop_p=scd_args.drop_p)

    if scd_args.resume:

        temp = torch.load(os.path.join(scd_args.save_path, scd_args.source + '.pt'),
                          map_location=torch.device('cpu'))
        print(f'Load state dict {scd_args.source} successfully')
        net.load_state_dict(temp)
        best_model.load_state_dict(net.state_dict())
    criterion = scd_args.criterion()

    if scd_args.cuda:
        print('start move to cuda')
        torch.cuda.manual_seed_all(scd_args.seed)
        # torch.backends.cudnn.deterministic = True
        cudnn.benchmark = True
        if scd_args.fp16:
            net = net.half()

        # net = torch.nn.DataParallel(net, device_ids=[0,1])
        device = torch.device("cuda:%s" % scd_args.gpu)
        net.to(device=device)
        embedding.to(device=device)
        best_model.to(device=device)
        criterion.to(device=device, dtype=dtype)

    net.eval()

    best_acc = 0

    # Training
    for epoch in range(scd_args.num_iters):

        print(f'\nEpoch: {epoch}')
        start_time = time.time()

        layers = net.layers[::-1]  # reverse the order of layers' name
        if epoch > scd_args.diversity_train_stop_iters:
            scd_args.fc_diversity = False
            scd_args.conv_diversity = False

        if scd_args.lr_decay_iter and (epoch + 1) % scd_args.lr_decay_iter == 0:
            scd_args.w_inc1 /= 2
            scd_args.w_inc2 /= 2
            print(f'w_inc1: {scd_args.w_inc1}, w_inc2: {scd_args.w_inc2}')

        if scd_args.batch_increase_iter and (epoch + 1) % scd_args.batch_increase_iter == 0:
            train_loader.nrows *= 2
            print(f'nrows: {train_loader.nrows}')
        # if scd_args.adaptive_loss_epoch:
        #     criterion = scd_args.criterion(kind='balanced')
        #     if scd_args.adaptive_loss_epoch < epoch:
        #         criterion = scd_args.criterion(kind='combined')
        with torch.no_grad():
            if epoch == 0 and not scd_args.resume:
                print(f'Randomly initialization based on {scd_args.init} distribution')
                # init_bias(net, data)
            # update Final layer
            # p = iter(train_loader)
            # data, target = p.next()
            for i, batch in enumerate(train_loader):
                # print(f'Iters #{i}')
                data = batch.text
                target = batch.label

                net.train()
            # for batch_idx, (data, target) in enumerate(test_loader):
                if use_cuda:
                    data, target = data.to(device=device), target.to(device=device)

                data = embedding(data)
                data.unsqueeze_(dim=1)
                # initial bias


                for layer_index, layer in enumerate(layers):
                    if len(layers) - layer_index <= scd_args.freeze_layer:
                        print(f'skip {layer} in training')
                        continue
                    # print(layer)
                    if 'fc' in layer:
                        if scd_args.no_bias == 2 or scd_args.no_bias and layer_index != 0:
                            update_mid_layer_fc_nobias(net, layers, layer_index, data, dtype,
                                                scd_args, criterion, target, device)
                        else:
                            update_mid_layer_fc(net, layers, layer_index, data, dtype,
                                                scd_args, criterion, target, device)
                        # continue
                    elif 'conv' in layer:
                        # continue
                        if scd_args.no_bias == 2 or scd_args.no_bias and layer_index != 0:
                            update_mid_layer_conv_nobias(net, layers, layer_index, data,
                                dtype, scd_args, criterion, target, device)
                        else:
                            update_mid_layer_conv(net, layers, layer_index, data,
                                                         dtype, scd_args, criterion, target, device)

                # print(net.fc2.weight)
            end_time = time.time() - start_time
            print('This epoch training cost {:.3f} seconds'.format(end_time))
            if (epoch + 1) % scd_args.verbose_iter == 0:
                val_acc = evaluation_text(val_loader, use_cuda, device,
                                     dtype, net, 'Train', embedding, criterion)
                if val_acc > best_acc:
                    best_acc = val_acc
                    print('New best acc: ', best_acc)
                    best_model.load_state_dict(net.state_dict())
                test_acc = evaluation_text(test_loader, use_cuda, device,
                                      dtype, net, 'Test', embedding, criterion)

                if scd_args.record:
                    temp_features = get_features(test_loader, use_cuda, device,
                                          dtype, net, 'Test')
                    if epoch == 0:
                        previous_features = temp_features
                        features = {}
                        for layer in layers:
                            features[layer] = []
                    else:
                        for layer in layers:
                            tf = temp_features[layer]
                            pf = previous_features[layer]
                            diff = (tf.view((tf.size(0), -1)) == pf.view((pf.size(0), -1))).float().mean(dim=1)

                            features[layer].append(diff.numpy())
                        previous_features = temp_features


                temp_row = pd.Series(
                    {'epoch': epoch + 1,
                     'train acc': val_acc,
                     'test acc': test_acc,
                     }
                )

                if scd_args.save:
                    if epoch == 0:
                        with open(log_file_name + '.temp', 'w') as f:
                            f.write('epoch, train_acc, test_acc\n')
                    else:
                        with open(log_file_name + '.temp', 'a') as f:
                            f.write(f'{epoch+1}, {val_acc}, {test_acc}\n')


                df = df.append(temp_row, ignore_index=True)

            if scd_args.temp_save_per_iter and (epoch + 1) % scd_args.temp_save_per_iter == 0:
                if not os.path.exists(scd_args.save_path):
                    os.makedirs(scd_args.save_path)
                torch.save(best_model.cpu().state_dict(),
                           os.path.join(scd_args.save_path,
                                        scd_args.target + f'_iter#{epoch+1}') + '.pt'
                           )
                print(f"Save {scd_args.target + f'_iter#{epoch+1}' + '.pt'} successfully")

    df.to_csv(log_file_name + '.csv', index=False)
    if scd_args.save:
        if not os.path.exists(scd_args.save_path):
            os.makedirs(scd_args.save_path)
        torch.save(best_model.cpu().state_dict(),
            os.path.join(scd_args.save_path, scd_args.target) + '.pt'
        )
        print(f"Save {scd_args.target + '.pt'} successfully")
        if scd_args.record:
            for name in scd_args.logs.keys():
                # scd_args.logs[name] = np.stack(scd_args.logs[name], axis=0)
                if scd_args.save:
                    dt = pd.DataFrame(scd_args.logs[name])
                    dt.to_csv(log_file_name + f'_{name}.csv', index=False)
            for layer in layers:
                fl = np.stack(features[layer], axis=1)
                dfs = {'data_index': np.arange(fl.shape[0])}
                for i in range(fl.shape[1]):
                    dfs[f'ep{i+1}'] = fl[:, i]

                pd.DataFrame(dfs).to_csv(log_file_name + f'_{layer}_features.csv', index=False)

    end_time_ = time.time()
    print('Cost %.1f seconds' % (end_time_ - start_time_))
    return best_model.cpu(), embedding.cpu(), best_acc, TEXT.vocab.stoi


if __name__ == '__main__':
    et, vc = print_title()
    scd_args = ModelArgs()

    scd_args.nrows = 300
    scd_args.nfeatures = 1
    scd_args.w_inc = 0.17
    scd_args.tol = 0.00000
    scd_args.local_iter = 1
    scd_args.num_iters = 50
    scd_args.interval = 10
    scd_args.rounds = 1
    scd_args.w_inc1 = 0.1
    scd_args.updated_fc_features = 128
    scd_args.updated_conv_features = 128
    scd_args.n_jobs = 1
    scd_args.num_gpus = 1
    scd_args.adv_train = False
    scd_args.eps = 0.1
    scd_args.w_inc2 = 0.1
    scd_args.hidden_nodes = 20
    scd_args.evaluation = True
    scd_args.verbose = True
    scd_args.b_ratio = 0.2
    scd_args.cuda = True
    scd_args.seed = 0
    scd_args.target = 'sms_wordcnn01'
    scd_args.source = 'sms_bp_d0_long_encoder_0'
    scd_args.save = False
    scd_args.resume = True
    scd_args.criterion = CrossEntropyLoss
    scd_args.structure = arch['wordcnn01']
    scd_args.dataset = 'sms'
    scd_args.num_classes = 2
    scd_args.gpu = 0
    scd_args.fp16 = False
    scd_args.act = 'sign'
    scd_args.updated_fc_nodes = 1
    scd_args.updated_conv_nodes = 10
    scd_args.width = 100
    scd_args.normal_noise = False
    scd_args.verbose = True
    scd_args.normalize = False
    scd_args.batch_size = 16
    scd_args.sigmoid = False
    scd_args.softmax = True
    scd_args.percentile = True
    scd_args.fail_count = 1
    scd_args.diversity = False
    scd_args.fc_diversity = False
    scd_args.conv_diversity = False
    scd_args.updated_conv_features_diversity = 3
    scd_args.diversity_train_stop_iters = 3000
    scd_args.updated_fc_ratio = 10
    scd_args.updated_conv_ratio = 10
    scd_args.init = 'normal'
    scd_args.logs = {}
    scd_args.no_bias = 2
    scd_args.record = False
    scd_args.scale = 1
    scd_args.save_path = os.path.join('../experiments/checkpoints', 'pt')
    scd_args.divmean = 0
    scd_args.cnn = 1
    scd_args.verbose_iter = 1
    scd_args.freeze_layer = 0
    scd_args.temp_save_per_iter = 0
    scd_args.lr_decay_iter = 0
    scd_args.batch_increase_iter = 0
    scd_args.aug = 0
    scd_args.balanced_sampling = 0
    scd_args.embedding_path = '../experiments/glove.6B.200d.txt'
    scd_args.drop_p = 0


    np.random.seed(scd_args.seed)

    data_set = [f'../data/{scd_args.dataset}/all_token.csv',
                f'../data/{scd_args.dataset}/train_token.csv',
                f'../data/{scd_args.dataset}/test_token.csv', ]

    best_model, embedding, val_acc, stoi = train_single_cnn01(
        scd_args, None, None, data_set)
    print('train acc: ', val_acc)
    model = TextWrapper(embedding_layer=embedding, model=best_model, stoi=stoi)

    # text = pd.read_csv(data_set[2])['text'].values.tolist()
    # text = ['"well', 'thats', 'nice.', 'too', 'bad', 'i', 'cant', 'eat', 'it']
    # print(model.text_pred([text]*6))

    with open(f'../experiments/checkpoints/{scd_args.target}.pkl', 'wb') as f:
        pickle.dump(model, f)


