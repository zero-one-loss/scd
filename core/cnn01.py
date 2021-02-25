import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


def sign(x):
    return x.sign_()
    # return torch.sigmoid(x)


def signb(x):
    return F.relu_(torch.sign(x)).float()
    # return x.float()

def softmax_(x):
    return F.softmax(x.float(), dim=1)

# def softmax_(x):
#     return x.float()


def sigmoid_(x):
    return torch.sigmoid(x)

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

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def _01_init(model):
    for name, m in model.named_modules():
        if 'si' not in name:
            if isinstance(m, nn.Conv2d):
                m.weight = torch.nn.Parameter(
                    m.weight.sign()
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight = torch.nn.Parameter(
                    m.weight.sign())
                if m.bias is not None:
                    m.bias.data.zero_()


def init_weights(model, kind='normal'):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(
                    m.weight.view((m.weight.size(0), -1)),
                    dim=1).view((-1, 1, 1, 1))
            )
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False

        if isinstance(m, nn.Linear):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(m.weight, dim=1, keepdim=True))
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False


class Toy3srr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3srr100, self).__init__()
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
            out = torch.sign(out)
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


class Toy3ssr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3ssr100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class Toy3sss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3sss100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class Toy2ss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy2ss100, self).__init__()
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
        self.fc3_si = nn.Linear(512, 100, bias=bias)
        self.fc4_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "fc3_si", "fc4_si"]
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
            out = torch.sign(out)
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
            out = torch.sign(out)
            out = F.avg_pool2d(out, 4)
            out = out.reshape(out.size(0), -1)
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
            out = torch.relu_(out)
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
            out = self.signb(out)

        return out


class Toy2sr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy2sr100, self).__init__()
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
        self.fc3_si = nn.Linear(512, 100, bias=bias)
        self.fc4_si = nn.Linear(100, num_classes, bias=bias)
        self.layers = ["conv1_si", "conv2_si", "fc3_si", "fc4_si"]
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
            out = torch.sign(out)
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
            out = F.avg_pool2d(out, 4)
            out = out.reshape(out.size(0), -1)
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
            out = torch.relu_(out)
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
            out = self.signb(out)

        return out


class Toy3rrs100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rrs100, self).__init__()
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
            out = torch.sign(out)
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


class Toy3rsr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rsr100, self).__init__()
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
            out = torch.sign(out)
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


class Toy3rss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rss100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class Toy3rrss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rrss100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class Toy3rsss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rsss100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class Toy3rsrs100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rsrs100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class WordCNN01(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False,
                 ndim=100, drop_p=0):
        super(WordCNN01, self).__init__()
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
        # if no_embed:
        #     self.embedding = None
        # else:
        #     self.embedding = nn.Embedding(nwords, ndim)
        self.conv1_si = nn.Conv2d(1, 150, kernel_size=(4, ndim), padding=(2, 0), bias=bias)
        self.fc2_si = nn.Linear(150, num_classes, bias=bias)
        self.layers = ["conv1_si", "fc2_si"]
        self.drop = nn.Dropout(drop_p)
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
            out = torch.sign(out)
            out.squeeze_(dim=-1)
            # out = F.avg_pool1d(out, out.size(2)).sign()
            out = F.relu(out).sum(dim=2)
            out = out.reshape(out.size(0), -1)
            # out = out / out.norm(dim=1, keepdim=True)
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.drop(out)
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            # out = out / out.norm(dim=1, keepdim=True)
            out = self.signb(out)

        return out


class toy3ssss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssss100, self).__init__()
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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
            out = torch.sign(out)
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


class toy3rrs100as(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3rrs100as, self).__init__()
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
            out = torch.sign(out)
            out = F.avg_pool2d(out, 2).sign()
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


class toy3rrss100as(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3rrss100as, self).__init__()
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
            out = torch.sign(out)
            out = F.avg_pool2d(out, 2).sign()
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
            out = torch.sign(out)
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


class toy3rrs100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3rrs100fs, self).__init__()
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class toy3rrss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3rrss100fs, self).__init__()
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
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

class Toy3rsr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rsr100fs, self).__init__()
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class Toy3rss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rss100fs, self).__init__()
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class Toy3rsss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(Toy3rsss100fs, self).__init__()
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            # out = F.avg_pool2d(out, 2).sign()
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
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

class toy3srr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3srr100fs, self).__init__()
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class toy3ssr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssr100fs, self).__init__()
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class toy3sss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3sss100fs, self).__init__()
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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


class toy3ssss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssss100fs, self).__init__()
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
            out = F.relu(out)
            out = F.avg_pool2d(out, 2) * 4
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
            out = torch.sign(out)
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


class toy3srr100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3srr100scale, self).__init__()
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
            out = self.signb(out)

        return out


class toy3ssr100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssr100scale, self).__init__()
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
            out = self.signb(out)

        return out


class toy3sss100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3sss100scale, self).__init__()
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
            out = self.signb(out)

        return out


class toy3ssss100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(toy3ssss100scale, self).__init__()
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
            out = self.signb(out)

        return out


class mlp01(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp01, self).__init__()
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
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlpsr, self).__init__()
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
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlprr, self).__init__()
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
            out = self.signb(out)

        return out


class mlp01scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
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
            self.signb = softmax_
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


arch = {}


arch['toy3srr100'] = Toy3srr100
arch['toy3ssr100'] = Toy3ssr100
arch['toy3sss100'] = Toy3sss100
arch['toy2ss100'] = Toy2ss100
arch['toy2sr100'] = Toy2sr100

arch['toy3rrs100'] = Toy3rrs100
arch['toy3rrr100'] = Toy3rrr100
arch['toy3rsr100'] = Toy3rsr100
arch['toy3rss100'] = Toy3rss100
arch['toy3rrss100'] = Toy3rrss100
arch['toy3rsss100'] = Toy3rsss100
arch['toy3rsrs100'] = Toy3rsrs100
arch['wordcnn01'] = WordCNN01
arch['toy3ssss100'] = toy3ssss100


arch['toy3rrs100as'] = toy3rrs100as
arch['toy3rrss100as'] = toy3rrss100as
arch['toy3rrs100fs'] = toy3rrs100fs
arch['toy3rrss100fs'] = toy3rrss100fs

arch['toy3rsr100fs'] = Toy3rsr100fs
arch['toy3rss100fs'] = Toy3rss100fs
arch['toy3rsss100fs'] = Toy3rsss100fs


arch['toy3srr100fs'] = toy3srr100fs
arch['toy3ssr100fs'] = toy3ssr100fs
arch['toy3sss100fs'] = toy3sss100fs
arch['toy3ssss100fs'] = toy3ssss100fs

arch['toy3srr100scale'] = toy3srr100scale
arch['toy3ssr100scale'] = toy3ssr100scale
arch['toy3sss100scale'] = toy3sss100scale
arch['toy3ssss100scale'] = toy3ssss100scale

arch['mlp01'] = mlp01
arch['mlpsr'] = mlpsr
arch['mlprr'] = mlprr


arch['mlp01scale'] = mlp01scale

arch['mlp2srscale'] = mlp2srscale
arch['mlp2ssscale'] = mlp2ssscale
arch['mlp2rr'] = mlp2rr
if __name__ == '__main__':
    # net = Toy(1, act='sign')
    # x = torch.rand(size=(100, 3, 32, 32))
    # # x = torch.rand(size=(100, 6, 16, 16))
    # output = net(x)  # shape 100, 1
    # Test case
    x = torch.rand(size=(8, 12)).long()
    # net = Toy2FConv(10, act='sign', sigmoid=False, softmax=True, scale=1, bias=True)
    net = arch['wordcnn01'](num_classes=2, act=sign, sigmoid=False, softmax=True,
                 no_embed=False, nwords=40, ndim=100, drop_p=0.3)
    #
    net.eval()
    output = net(x)
    layers = net.layers
    temp_out = x
    for i in range(len(layers)):
        print(f'Running on {layers[i]}')
        out = net(temp_out, input_=layers[i])
        temp_projection = net(temp_out, input_=layers[i], layer=layers[i] + '_projection')
        current_out = net(temp_out, input_=layers[i], layer=layers[i] + '_output')
        temp_out = net(temp_projection, input_=layers[i] + '_ap', layer=layers[i] + '_output')
    # import os
    # path = [os.path.join('checkpoints', 'toy_v3.pt') for i in range(200)]
    # net = Ensemble(structure=Toy(1, 'sign', False, False), path=path)
    # yp = net.predict_proba(x)
