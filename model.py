from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Normalize(nn.Module):
    def __init__(self, mean, std, device='cuda'):
        super().__init__()
        self.mean = torch.tensor(mean, requires_grad=False).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std, requires_grad=False).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std


class WideResNetBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WideResNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideResNetBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(WideResNetBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=16, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WideResNetBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = WideResNetBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = WideResNetBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = WideResNetBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                # init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                tmp = np.sqrt(3. / m.weight.data.shape[0])
                m.weight.data.uniform_(-tmp, tmp)
                m.bias.data.zero_()
                # init.kaiming_normal_(m.weight)
                # init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wideresnet16(**kwargs):
    return WideResNet(depth=16, **kwargs)


def wideresnet22(**kwargs):
    return WideResNet(depth=22, **kwargs)


class MnistModel(nn.Module):
    """ Construct basic MnistModel for mnist adversal attack """

    def __init__(self, re_init=False, has_dropout=False):
        super(MnistModel, self).__init__()
        self.re_init = re_init
        self.has_dropout = has_dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        if self.has_dropout:
            self.dropout = nn.Dropout()

        if self.re_init:
            self._init_params(self.conv1)
            self._init_params(self.conv2)
            self._init_params(self.fc1)
            self._init_params(self.fc2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        if self.has_dropout:
            x = self.dropout(x)

        x = self.fc2(x)

        return x

    def _init_params(self, module, mean=0.1, std=0.1):
        init.normal_(module.weight, std=0.1)
        if hasattr(module, 'bias'):
            init.constant_(module.bias, mean)


class ConvNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', norm='instancenorm',
                 net_pooling='avgpooling', im_size=(32, 32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, output_feat=False):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if output_feat:
            return out
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(shape_feat[0] // 16, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


class ConvNetFRePo(nn.Module):
    def __init__(self, channel=3, num_classes=10, net_width=128, net_depth=3, net_act='relu', norm='batchnorm',
                 net_pooling='avgpooling', im_size=(32, 32)):
        super(ConvNetFRePo, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, norm, net_act, net_pooling,
                                                      im_size)
        num_feat = shape_feat[0] * shape_feat[1] * shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x, output_feat=False):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        if output_feat:
            return out
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(shape_feat[0] // 16, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [
                nn.Conv2d(in_channels, 2 ** d * net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = 2 ** d * net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = 2 ** d * net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


''' ResNet '''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='batchnorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(planes, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes, planes, affine=True)
            bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes // 16, planes, affine=True)
            bn3 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            bn3 = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                bn3
            )
        self.skip = False

    def forward(self, x):
        if self.skip:
            return F.relu(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def drop_path(x, keep_prob: float = 1.0, inplace: bool = False):
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x


class DropPath(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class BasicBlockDP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='batchnorm', p=1):
        super(BasicBlockDP, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(planes, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes, planes, affine=True)
            bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes // 16, planes, affine=True)
            bn3 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            bn3 = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.droppath = DropPath(p=p)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1),
                # nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                bn3
            )

    def forward(self, x):
        feat = F.relu(self.bn1(self.conv1(x)))
        feat = self.bn2(self.conv2(feat))
        feat = self.droppath(feat)
        out = feat + self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockDP_DualNorm(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, norm='batchnorm', p=1):
        super(BasicBlockDP_DualNorm, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(planes, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        # normalization for adversarial examples
        if self.norm == 'instancenorm':
            self.bn1_adv = nn.GroupNorm(planes, planes, affine=True)
            self.bn2_adv = nn.GroupNorm(planes, planes, affine=True)
            self.bn3_adv = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1_adv = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2_adv = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn3_adv = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1_adv = nn.BatchNorm2d(planes)
            self.bn2_adv = nn.BatchNorm2d(planes)
            self.bn3_adv = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.droppath = DropPath(p=p)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1),
                # nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                # self.bn3
            )

        self.adv = False

    def forward(self, x):
        if not self.adv:
            feat = F.relu(self.bn1(self.conv1(x)))
            feat = self.bn2(self.conv2(feat))
            feat = self.droppath(feat)
            out = feat + self.bn3(self.shortcut(x))
            out = F.relu(out)
        else:
            feat = F.relu(self.bn1_adv(self.conv1(x)))
            feat = self.bn2_adv(self.conv2(feat))
            feat = self.droppath(feat)
            out = feat + self.bn3_adv(self.shortcut(x))
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(planes, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
            bn4 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
            bn4 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            bn4 = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                bn4
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckDP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='batchnorm', p=1):
        super(BottleneckDP, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(planes, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
            bn4 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn2 = nn.GroupNorm(planes // 16, planes, affine=True)
            self.bn3 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
            bn4 = nn.GroupNorm(self.expansion * planes // 16, self.expansion * planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            bn4 = nn.BatchNorm2d(self.expansion * planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.droppath = DropPath(p=p)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1),
                bn4
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.droppath(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetDP(nn.Module):
    def __init__(self, block=BasicBlockDP, num_blocks=[2, 2, 2, 2], channel=3, num_classes=10, norm='batchnorm',
                 im_size=32):
        super(ResNetDP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, self.in_planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(p=0.)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        if output_feat:
            return out
        out = self.classifier(out)
        return out


class ResNetDP_DualNorm(nn.Module):
    def __init__(self, block=BasicBlockDP_DualNorm, num_blocks=[2, 2, 2, 2], channel=3, num_classes=10, norm='batchnorm',
                 im_size=32):
        super(ResNetDP_DualNorm, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, self.in_planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        if self.norm == 'instancenorm':
            self.bn1_adv = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1_adv = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1_adv = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(p=0.)

        self.adv = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_feat=False):
        out = F.relu(self.bn1(self.conv1(x))) if not self.adv else F.relu(self.bn1_adv(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        if output_feat:
            return out
        out = self.classifier(out)
        return out


class ResNetProgressive(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), channel=3, num_classes=10, norm='batchnorm'):
        super(ResNetProgressive, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, self.in_planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')
        self.layers = []
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))
        self.layers = nn.ModuleList(self.layers)
        self.classifier = nn.Linear(512, num_classes)
        self.step = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.avg_pool2d(out, 2)
        for i, layer in enumerate(self.layers):
            # if i < self.step:
            #     set_skip(layer, skip=False)
            # else:
            #     set_skip(layer, skip=True)
            out = layer(out)
            if i + 1 == self.step:
                break
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if output_feat:
            return out
        out = self.classifier(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm
        self.conv1 = nn.Conv2d(channel, self.in_planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if output_feat:
            return out
        out = self.classifier(out)
        return out


class ConvNetSC(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=1, channel=3, num_classes=10, norm='batchnorm'):
        super(ConvNetSC, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, self.in_planes, kernel_size=3, stride=1, padding=1)
        if self.norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(self.in_planes, self.in_planes, affine=True)
        elif self.norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(self.in_planes // 16, self.in_planes, affine=True)
        elif self.norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(self.in_planes)
        else:
            raise ValueError(f'no {self.norm} layer')
        self.layers = []
        self.layers.append(self._make_layer(block, 128, num_blocks, stride=1))
        self.layers = nn.ModuleList(self.layers)
        self.classifier = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.avg_pool2d(out, 2)
        for i, layer in enumerate(self.layers):
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if output_feat:
            return out
        out = self.classifier(out)
        return out


def ResNet18(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm=norm)


def ResNet34(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes, norm=norm)


def ResNet50(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes, norm=norm)

def ResNet50DP(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(BottleneckDP, [3, 4, 6, 3], channel=channel, num_classes=num_classes, norm=norm)


def ResNet101(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes, norm=norm)


def ResNet152(channel, num_classes, norm='batchnorm', im_size=32):
    return ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes, norm=norm)


''' AlexNet '''
class ConvDP(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel, kernel_size1, kernel_size2, norm, pool=None, p=1):
        super(ConvDP, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size1, padding=(kernel_size1-1)//2)
        self.conv2 = nn.Conv2d(hidden_channel, output_channel, kernel_size=kernel_size2, padding=(kernel_size2-1)//2)
        if norm == 'instancenorm':
            self.bn1 = nn.GroupNorm(hidden_channel, hidden_channel, affine=True)
            self.bn2 = nn.GroupNorm(output_channel, output_channel, affine=True)
            self.bn = nn.GroupNorm(output_channel, output_channel, affine=True)
        elif norm == 'groupnorm':
            self.bn1 = nn.GroupNorm(hidden_channel // 16, hidden_channel, affine=True)
            self.bn2 = nn.GroupNorm(output_channel // 16, output_channel, affine=True)
            self.bn = nn.GroupNorm(output_channel // 16, output_channel, affine=True)
        elif norm == 'batchnorm':
            self.bn1 = nn.BatchNorm2d(hidden_channel)
            self.bn2 = nn.BatchNorm2d(output_channel)
            self.bn = nn.BatchNorm2d(output_channel)
        else:
            raise ValueError(f'no {norm} layer')

        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        if input_channel == output_channel:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Identity() if pool is None else nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(input_channel, output_channel, kernel_size=1),
                self.bn
            )

        self.p = p

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.pool is not None:
            out = self.pool(out)
        out = F.relu(self.bn2(self.conv2(out)))

        if self.training and self.p > 0:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
            mask = x.new_empty(mask_shape).bernoulli_(self.p)
            out = out * mask + self.shortcut(x) * (1 - mask)

        return out


class AlexNet(nn.Module):
    def __init__(self, channel, num_classes, norm='batchnorm', im_size=(32, 32)):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2),
            nn.BatchNorm2d(128) if norm == 'batchnorm' else nn.GroupNorm(128, 128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192) if norm == 'batchnorm' else nn.GroupNorm(192, 192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if norm == 'batchnorm' else nn.GroupNorm(256, 256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192) if norm == 'batchnorm' else nn.GroupNorm(192, 192, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192) if norm == 'batchnorm' else nn.GroupNorm(192, 192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * im_size[0]//8 * im_size[1]//8, num_classes)

    def forward(self, x, output_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if output_feat:
            return x
        x = self.fc(x)
        return x


class AlexNetDP(nn.Module):
    def __init__(self, channel, num_classes, norm, p=1, im_size=(32, 32)):
        super(AlexNetDP, self).__init__()
        self.p = p
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 192, kernel_size=5, padding=2),
            # nn.BatchNorm2d(192),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(192, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            ConvDP(128, 192, 256, 5, 3, norm=norm, pool='max', p=p),
            # nn.Conv2d(256, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(inplace=True),
            ConvDP(256, 192, 192, 3, 3, norm=norm, p=p),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * im_size[0]//8 * im_size[1]//8, num_classes)

    def forward(self, x, output_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if output_feat:
            return x
        x = self.fc(x)
        return x


''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='batchnorm', im_size=(32, 32)):
        super(VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
        self.classifier = nn.Linear(512*im_size[0]//32*im_size[1]//32 if vgg_name != 'VGGS' else 128*im_size[1]//32*im_size[1]//32, num_classes)

    def forward(self, x, output_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if output_feat:
            return x
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11DP(nn.Module):
    def __init__(self, channel, num_classes, norm='batchnorm', p=1, im_size=(32, 32)):
        super(VGG11DP, self).__init__()
        self.channel = channel
        self.norm = norm
        self.p = p
        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, padding=3 if self.channel == 1 else 1),
            nn.GroupNorm(64, 64, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(128, 128, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvDP(128, 256, 256, 3, 3, norm=norm, p=p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvDP(256, 512, 512, 3, 3, norm=norm, p=p),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvDP(512, 512, 512, 3, 3, norm=norm, p=p),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(512*im_size[0]//32*im_size[1]//32, num_classes)

    def forward(self, x, output_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if output_feat:
            return x
        x = self.classifier(x)
        return x


def VGG11(channel, num_classes, norm, im_size):
    return VGG('VGG11', channel, num_classes, norm, im_size=im_size)


def VGG13(channel, num_classes, norm):
    return VGG('VGG13', channel, num_classes, norm)


def VGG16(channel, num_classes, norm):
    return VGG('VGG16', channel, num_classes, norm)


def VGG19(channel, num_classes, norm):
    return VGG('VGG19', channel, num_classes, norm)


__factory = {
    # resnet series, kwargs: num_classes
    'resnet': ResNet18,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet50dp': ResNet50DP,
    # wideresnet series, kwargs: num_classes, widen_factor, dropRate
    'wide': wideresnet16,
    'wideresnet': wideresnet16,
    'wideresnet16': wideresnet16,
    'wideresnet22': wideresnet22,
    # mnist, kwargs: has_dropout
    'mnist': MnistModel,
    'convnet': ConvNet,
    'convnetfrepo': ConvNetFRePo,
    'resnetdp': ResNetDP,
    'resnetdp_dualnorm': ResNetDP_DualNorm,
    'resnetprog': ResNetProgressive,
    'convnetsc': ConvNetSC,
    'vgg': VGG11,
    'alexnet': AlexNet,
    'vggdp': VGG11DP,
    'alexnetdp': AlexNetDP,
}


def create_model(name, **kwargs):
    assert (name in __factory), 'invalid network'
    return __factory[name](**kwargs)


def set_drop_path(model, p):
    if isinstance(model, (DropPath, ConvDP)):
        model.p = p
    else:
        for n, m in model.named_modules():
            if isinstance(model, (DropPath, ConvDP)):
                m.p = p


def set_skip(model, skip):
    if isinstance(model, BasicBlock):
        model.skip = skip
    else:
        for m in model.modules():
            if isinstance(m, BasicBlock):
                m.skip = skip


def set_dual_norm(model, adv):
    if isinstance(model, (BasicBlockDP_DualNorm, ResNetDP_DualNorm)):
        model.adv = adv
    for m in model.modules():
        if isinstance(m, BasicBlockDP_DualNorm):
            m.adv = adv


if __name__ == '__main__':
    net = create_model('resnetdp', num_classes=10)
    # import pdb; pdb.set_trace()  # breakpoint 2e2204d9 //
    set_drop_path(net, 0.5)
