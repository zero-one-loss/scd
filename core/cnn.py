import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def softmax_(x):
    return F.softmax(x.float(), dim=1)


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
           init.constant_(m.bias, 0)
           
# def weights_init(m):
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
#     if isinstance(m, nn.Linear):
#         init.kaiming_normal_(m.weight, mode='fan_in')
#         if m.bias is not None:
#             init.constant_(m.bias, 0)


class MySign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        # scale = math.sqrt(inputs.size(1)) * 3
        grad_output[inputs.abs()>inputs.abs().mean()/4] = 0
        return grad_output

msign = MySign.apply


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
    def __init__(self, num_classes=10, scale=1, bias=False):
        super(Cifar10CNN, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16 * scale, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * scale, 16 * scale, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(16 * scale, 32 * scale, 3, padding=1, bias=bias)
        self.conv4_si = nn.Conv2d(32 * scale, 32 * scale, 3, padding=1, bias=bias)
        self.conv5_si = nn.Conv2d(32 * scale, 64 * scale, 3, padding=1, bias=bias)
        self.conv6_si = nn.Conv2d(64 * scale, 64 * scale, 3, padding=1, bias=bias)
        self.conv7_si = nn.Conv2d(64 * scale, 128 * scale, 3, padding=1, bias=bias)
        self.conv8_si = nn.Conv2d(128 * scale, 128 * scale, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(16 * scale)
        self.bn2 = nn.BatchNorm2d(16 * scale)
        self.bn3 = nn.BatchNorm2d(32 * scale)
        self.bn4 = nn.BatchNorm2d(32 * scale, )
        self.bn5 = nn.BatchNorm2d(64 * scale, )
        self.bn6 = nn.BatchNorm2d(64 * scale, )
        self.bn7 = nn.BatchNorm2d(128 * scale, )
        self.bn8 = nn.BatchNorm2d(128 * scale, )
        self.fc9_si = nn.Linear(128 * scale, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.conv1_si(x))
        out = F.relu(self.conv2_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv3_si(out))
        out = F.relu(self.conv4_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv5_si(out))
        out = F.relu(self.conv6_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv7_si(out))
        out = F.relu(self.conv8_si(out))
        out = F.avg_pool2d(out, 4)
        out = out.reshape((out.size(0), -1))
        out = self.fc9_si(out)
        return out

class Cifar10CNNbn(nn.Module):
    def __init__(self, num_classes=10, scale=1, bias=False):
        super(Cifar10CNNbn, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16 * scale, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * scale, 16 * scale, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(16 * scale, 32 * scale, 3, padding=1, bias=bias)
        self.conv4_si = nn.Conv2d(32 * scale, 32 * scale, 3, padding=1, bias=bias)
        self.conv5_si = nn.Conv2d(32 * scale, 64 * scale, 3, padding=1, bias=bias)
        self.conv6_si = nn.Conv2d(64 * scale, 64 * scale, 3, padding=1, bias=bias)
        self.conv7_si = nn.Conv2d(64 * scale, 128 * scale, 3, padding=1, bias=bias)
        self.conv8_si = nn.Conv2d(128 * scale, 128 * scale, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(16 * scale)
        self.bn2 = nn.BatchNorm2d(16 * scale)
        self.bn3 = nn.BatchNorm2d(32 * scale)
        self.bn4 = nn.BatchNorm2d(32 * scale, )
        self.bn5 = nn.BatchNorm2d(64 * scale, )
        self.bn6 = nn.BatchNorm2d(64 * scale, )
        self.bn7 = nn.BatchNorm2d(128 * scale, )
        self.bn8 = nn.BatchNorm2d(128 * scale, )
        self.fc9_si = nn.Linear(128 * scale, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1_si(x)))
        out = F.relu(self.bn2(self.conv2_si(out)))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3_si(out)))
        out = F.relu(self.bn4(self.conv4_si(out)))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.bn5(self.conv5_si(out)))
        out = F.relu(self.bn6(self.conv6_si(out)))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.bn7(self.conv7_si(out)))
        out = F.relu(self.bn8(self.conv8_si(out)))
        out = F.avg_pool2d(out, 4)
        out = out.reshape((out.size(0), -1))
        out = self.fc9_si(out)
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

class Toy3ap2(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(Toy3ap2, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.conv1_si(x)
        out = F.avg_pool2d(out, 2)
        out = F.relu(out)
        out = self.conv2_si(out)
        out = F.avg_pool2d(out, 2)
        out = F.relu(out)
        out = F.relu(self.conv3_si(out))
        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc4_si(out))
        out = self.fc5_si(out)
        return out

class Toy3ap1(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(Toy3ap1, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.conv1_si(x)
        out = F.avg_pool2d(out, 2)
        out = F.relu(out)
        out = F.relu(self.conv2_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv3_si(out))
        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc4_si(out))
        out = self.fc5_si(out)
        return out

class Toy3sigmoid(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(Toy3sigmoid, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = torch.sigmoid(self.conv1_si(x))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv2_si(out))
        out = F.avg_pool2d(out, 2)
        out = F.relu(self.conv3_si(out))
        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc4_si(out))
        out = self.fc5_si(out)
        return out


class Toy3tanh(nn.Module):
    def __init__(self, num_classes=10, bias=True):
        super(Toy3tanh, self).__init__()
        self.conv1_si = nn.Conv2d(3, 16, 3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(64 * 4 * 4, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):
        out = torch.tanh(self.conv1_si(x))
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

class toy3srr100scale(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3srr100scale, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = msign
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(1024, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.0833
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.conv2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.conv3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            out = out.reshape(out.size(0), -1)
            if layer == self.layers[2] + "_output":
                return out

        # layer 4
        if input_ == self.layers[3]:
            out = x
        if status < 4:
            if input_ != self.layers[3] + "_ap":
                out = self.fc4_si(out)
            if layer == self.layers[3] + "_projection":
                return out
            if input_ == self.layers[3] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[3] + "_output":
                return out

        # layer 5
        if input_ == self.layers[4]:
            out = x
        if status < 5:
            if input_ != self.layers[4] + "_ap":
                out = self.fc5_si(out)
            if layer == self.layers[4] + "_projection":
                return out
            if input_ == self.layers[4] + "_ap":
                out = x
            # out = self.signb(out)

        return out


class toy3ssr100scale(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssr100scale, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = msign
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(1024, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.0833
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.conv2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = msign(out) * 0.0589
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.conv3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            out = out.reshape(out.size(0), -1)
            if layer == self.layers[2] + "_output":
                return out

        # layer 4
        if input_ == self.layers[3]:
            out = x
        if status < 4:
            if input_ != self.layers[3] + "_ap":
                out = self.fc4_si(out)
            if layer == self.layers[3] + "_projection":
                return out
            if input_ == self.layers[3] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[3] + "_output":
                return out

        # layer 5
        if input_ == self.layers[4]:
            out = x
        if status < 5:
            if input_ != self.layers[4] + "_ap":
                out = self.fc5_si(out)
            if layer == self.layers[4] + "_projection":
                return out
            if input_ == self.layers[4] + "_ap":
                out = x
            # out = self.signb(out)

        return out


class toy3sss100scale(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3sss100scale, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = msign
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(1024, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.0833
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.conv2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = msign(out) * 0.0589
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.conv3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = msign(out) * 0.0417
            out = F.avg_pool2d(out, 2)
            out = out.reshape(out.size(0), -1)
            if layer == self.layers[2] + "_output":
                return out

        # layer 4
        if input_ == self.layers[3]:
            out = x
        if status < 4:
            if input_ != self.layers[3] + "_ap":
                out = self.fc4_si(out)
            if layer == self.layers[3] + "_projection":
                return out
            if input_ == self.layers[3] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[3] + "_output":
                return out

        # layer 5
        if input_ == self.layers[4]:
            out = x
        if status < 5:
            if input_ != self.layers[4] + "_ap":
                out = self.fc5_si(out)
            if layer == self.layers[4] + "_projection":
                return out
            if input_ == self.layers[4] + "_ap":
                out = x
            # out = self.signb(out)

        return out


class toy3ssss100scale(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssss100scale, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = msign
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(1024, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.0833
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.conv2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = msign(out) * 0.0589
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.conv3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = msign(out) * 0.0417
            out = F.avg_pool2d(out, 2)
            out = out.reshape(out.size(0), -1)
            if layer == self.layers[2] + "_output":
                return out

        # layer 4
        if input_ == self.layers[3]:
            out = x
        if status < 4:
            if input_ != self.layers[3] + "_ap":
                out = self.fc4_si(out)
            if layer == self.layers[3] + "_projection":
                return out
            if input_ == self.layers[3] + "_ap":
                out = x
            out = msign(out) * 0.1000
            if layer == self.layers[3] + "_output":
                return out

        # layer 5
        if input_ == self.layers[4]:
            out = x
        if status < 5:
            if input_ != self.layers[4] + "_ap":
                out = self.fc5_si(out)
            if layer == self.layers[4] + "_projection":
                return out
            if input_ == self.layers[4] + "_ap":
                out = x
            # out = self.signb(out)

        return out

class mlp01(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp01, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = self.signb(out)

        return out

class mlpsr(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlpsr, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.sigmoid(out)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = self.signb(out)

        return out


class mlprr(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlprr, self).__init__()
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.relu(out)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = self.signb(out)

        return out


class mlp01scale(nn.Module):
    def __init__(self, num_classes=2, act='sign', sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp01scale, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = F.softmax
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.2236
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = self.signb(out)

        return out


class mlp2srscale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp2srscale, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, 20, bias=bias)
        self.fc3_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.2236
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.fc3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = self.signb(out)

        return out


class mlp2ssscale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp2ssscale, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, 20, bias=bias)
        self.fc3_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.2236
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = msign(out) * 0.2236
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.fc3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = self.signb(out)

        return out


class mlp2rr(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp2rr, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, 20, bias=bias)
        self.fc3_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.fc3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = self.signb(out)

        return out


class Toy3rrr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rrr100, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.conv3_si = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=bias)
        self.fc4_si = nn.Linear(1024, 100, bias=bias)
        self.fc5_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.conv1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.conv2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            if layer == self.layers[1] + "_output":
                return out

        # layer 3
        if input_ == self.layers[2]:
            out = x
        if status < 3:
            if input_ != self.layers[2] + "_ap":
                out = self.conv3_si(out)
            if layer == self.layers[2] + "_projection":
                return out
            if input_ == self.layers[2] + "_ap":
                out = x
            out = torch.relu_(out)
            out = F.avg_pool2d(out, 2)
            out = out.reshape(out.size(0), -1)
            if layer == self.layers[2] + "_output":
                return out

        # layer 4
        if input_ == self.layers[3]:
            out = x
        if status < 4:
            if input_ != self.layers[3] + "_ap":
                out = self.fc4_si(out)
            if layer == self.layers[3] + "_projection":
                return out
            if input_ == self.layers[3] + "_ap":
                out = x
            out = torch.relu_(out)
            if layer == self.layers[3] + "_output":
                return out

        # layer 5
        if input_ == self.layers[4]:
            out = x
        if status < 5:
            if input_ != self.layers[4] + "_ap":
                out = self.fc5_si(out)
            if layer == self.layers[4] + "_projection":
                return out
            if input_ == self.layers[4] + "_ap":
                out = x
            out = self.signb(out)

        return out

    
arch = {}

arch['toy3srr100scale'] = toy3srr100scale
arch['toy3ssr100scale'] = toy3ssr100scale
arch['toy3sss100scale'] = toy3sss100scale
arch['toy3ssss100scale'] = toy3ssss100scale
arch['toy3rrr100'] = Toy3rrr100
arch['mlpsr'] = mlpsr
arch['mlprr'] = mlprr
arch['mlp01'] = mlp01
arch['mlp01scale'] = mlp01scale
arch['mlp2srscale'] = mlp2srscale
arch['mlp2ssscale'] = mlp2ssscale
arch['mlp2rr'] = mlp2rr


if __name__ == '__main__':
    # net = Cifar10CNN2(2)
    x = torch.randn((100, 3, 32, 32))
    net = Toy3(2)
    output = net(x)