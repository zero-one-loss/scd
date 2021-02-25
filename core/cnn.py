import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
           init.constant_(m.bias, 0)
           

# def _weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.normal_(m.weight, mean=0, std=1)
#         # nn.init.uniform_(m.weight, -1, 1)
#         m.weight = torch.nn.Parameter(
#             m.weight / torch.norm(
#                 m.weight.view((m.weight.size(0), -1)),
#                 dim=1).view((-1, 1, 1, 1))
#         )
#         # m.weight.requires_grad = False
#         if m.bias is not None:
#             init.constant_(m.bias, 0)
#             # m.bias.requires_grad = False
#
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=1)
#         # nn.init.uniform_(m.weight, -1, 1)
#         # nn.init.zeros_(m.weight)
#         m.weight = torch.nn.Parameter(
#             m.weight / torch.norm(m.weight, dim=1, keepdim=True))
#         # m.weight.requires_grad = False
#         if m.bias is not None:
#             m.bias.data.zero_()
#             # m.bias.requires_grad = False
            
class Cifar10CNN1(nn.Module):
    def __init__(self, num_classes=2):
        super(Cifar10CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.conv1_ds = nn.Conv2d(8, 16, 2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.conv2_ds = nn.Conv2d(16, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.conv3_ds = nn.Conv2d(32, 64, 2, stride=2)
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv1_ds(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv2_ds(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv3_ds(out))
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        out = out.view((out.size(0), out.size(1)))
        out = self.fc(out)

        return out

class Cifar10CNN2(nn.Module):
    def __init__(self, num_classes=2):
        super(Cifar10CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1, bias=True)
        self.conv1_ds = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, bias=True)
        self.conv2_ds = nn.AvgPool2d(kernel_size=2, stride=4)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        # self.conv3_ds = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv1_ds(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_ds(out)
        out = F.relu(self.conv3(out))
        # out = self.conv3_ds(out)
        out = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        out = out.view((out.size(0), out.size(1)))
        out = self.fc(out)

        return out


class LeNet_cifar(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        # x = F.pad(x, 2)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10, scale=1):
        super(Cifar10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16 * scale, 3, padding=1)
        self.conv2 = nn.Conv2d(16 * scale, 16 * scale, 3, padding=1)
        self.conv3 = nn.Conv2d(16 * scale, 32 * scale, 3, padding=1)
        self.conv4 = nn.Conv2d(32 * scale, 32 * scale, 3, padding=1)
        self.conv5 = nn.Conv2d(32 * scale, 64 * scale, 3, padding=1)
        self.conv6 = nn.Conv2d(64 * scale, 64 * scale, 3, padding=1)
        # self.conv7 = nn.Conv2d(64 * scale, 128 * scale, 3, padding=1)
        # self.conv8 = nn.Conv2d(128 * scale, 128 * scale, 3, padding=1)
        self.fc = nn.Linear(64 * 16 * scale, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        # x = F.pad(x, 2)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv7(out))
        # out = F.relu(self.conv8(out))
        # out = F.avg_pool2d(out, out.size(2))
        out = out.reshape((out.size(0), -1))
        out = self.fc(out)
        return out

class Toy(nn.Module):
    def __init__(self, num_classes=2):
        super(Toy, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.fc1 = nn.Linear(6 * 8 * 8, 20)
        self.fc2 = nn.Linear(20, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class Toy2(nn.Module):
    def __init__(self, num_classes=10, scale=2):
        super(Toy2, self).__init__()
        self.conv1 = nn.Conv2d(3, 8 * scale, 3, padding=1)
        self.conv2 = nn.Conv2d(8 * scale, 16 * scale, 3, padding=1)
        self.fc1 = nn.Linear(16 * scale * 4 * 4, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class Toy3(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(Toy3, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1_si(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv3_si(out))
        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc4_si(out))
        out = self.fc5_si(out)
        return out


class Toy3BN(nn.Module):
    def __init__(self, num_classes=10):
        super(Toy3BN, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100)
        self.fc5_si = nn.Linear(100, num_classes)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-4,)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-4,)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-4, )
        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_si(x)))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2_si(out)))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3_si(out)))
        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc4_si(out))
        out = self.fc5_si(out)
        return out


if __name__ == '__main__':
    # net = Cifar10CNN2(2)
    x = torch.randn((100, 3, 32, 32))
    net = Toy3(2)
    output = net(x)