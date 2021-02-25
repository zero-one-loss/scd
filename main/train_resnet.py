import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import sys
sys.path.append('..')
from core.cnn import Toy2, Toy3, Toy3BN
from core.resnet_nob import resnet18
from tools import args
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from util.misc import CSVLogger



DATA = 'cifar10'

if DATA == 'cifar10':
    train_dir = '/home/y/yx277/research/ImageDataset/cifar10'
    test_dir = '/home/y/yx277/research/ImageDataset/cifar10'



resume = False
use_cuda = True
fp16 = True if args.fp16 else False
dtype = torch.float16 if fp16 else torch.float32



best_acc = 0

batch_size = 128
aug = args.aug
seed = args.seed
print('Random seed: ', seed)
torch.manual_seed(seed)
save_path = 'checkpoints/pt'
if not os.path.isdir('logs'):
    os.mkdir('logs')
filename = f'logs/cifar10/{args.target}.csv'
csv_logger = CSVLogger(fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

train_transform = transforms.Compose(
                [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.RandomApply([
                #     transforms.Lambda(
                #         lambda x: torch.clamp(x + torch.randn_like(x) * args.epsilon, min=0, max=1))],
                #     p=0.5)
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])

test_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])

if aug == 'noaug':
    train_transform = test_transform

print('start normalize')

trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# net = Toy3BN(num_classes=10)
net = resnet18(num_classes=10,)

criterion = nn.CrossEntropyLoss()

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])

if use_cuda:
    print('start move to cuda')
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    if fp16:
        net = net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    # net = torch.nn.DataParallel(net, device_ids=[0,1])
    device = torch.device("cuda:0")
    net.to(device=device)
    # criterion.to(device=device, dtype=dtype)




optimizer = optim.SGD(
    net.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)
# optimizer = optim.Adam(net.parameters(), lr=0.001, eps=1e-4 if fp16 else 1e-8)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    # global monitor
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()
    #    pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        outputs = net(data).float()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.size(0)
        predicted = outputs.max(1)[1]
        correct += predicted.eq(target).sum().item()

    print('Train loss: %0.5f,     Train_accuracy: %0.5f' % (
        train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' % (time.time() - a))

    return correct / len(train_loader.dataset)


def test(epoch):
    global best_acc
    # monitor
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs= net(data)
            loss = criterion(outputs, target)

            test_loss += loss.item() * target.size(0)
            predicted = outputs.max(1)[1]
            correct += predicted.eq(target).sum().item()

        print('Test loss: %0.5f,     Test_accuracy: %0.5f' % (
            test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
        print('This epoch cost %0.2f seconds' % (time.time() - a))

    acc = correct / len(test_loader.dataset)
    if acc > best_acc:
        print('Saving...')
        # state = {
        #     # 'net': net.module.state_dict(),
        #     'net': net.state_dict(),
        # }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(net.state_dict(), os.path.join(save_path, f'{args.target}.pt'))
        best_acc = acc

    return acc

def save_features():
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    try:
        os.mkdir('features0')
        os.chdir('features0')
    except:
        os.chdir('features0')

    train_data = np.zeros(shape=(8000, 512), dtype=np.float16)
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs, features = net(data)
            train_data[batch_idx*batch_size:(batch_idx+1)*batch_size] = features.cpu()
    np.save('test_data_20.npy', train_data)


def main():
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + 200):

        row = {'epoch': str(epoch), 'train_acc': str(train(epoch)), 'test_acc': str(test(epoch))}
        csv_logger.writerow(row)

        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        # if epoch in [60, 120, 160]:
        #     optimizer.param_groups[0]['lr'] *= 0.1
        scheduler.step()

def predict():
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    try:
        os.mkdir('features0')
        os.chdir('features0')
    except:
        os.chdir('features0')

    train_data = np.zeros(shape=(10000, ), dtype=np.int16)
    train_prob = np.zeros(shape=(10000, 10), dtype=np.float16)
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs= net(data)
            train_data[batch_idx*batch_size:(batch_idx+1)*batch_size] = outputs.max(1)[1]
            train_prob[batch_idx*batch_size:(batch_idx+1)*batch_size] = outputs.cpu()
    np.save('resnet_%s.npy'%sys.argv[1], train_data)
    np.save('resnet%s_prob.npy'%sys.argv[1], train_prob)

def inference():
    test(0)

if __name__ == '__main__':
    main()

