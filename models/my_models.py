from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torchvision.models import resnet152
from models.wpt_net import wpt_resnet_50, wpt_resnet_18, wpt_resnet_152


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class WPTResNet(models.ResNet):
    def __init__(self, input_channels, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

        if input_channels < 64:
            out_ch = self.layer1[0].conv1.out_channels
            kernel_size = self.layer1[0].conv1.kernel_size
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=kernel_size, stride=1,
                                   bias=False)
            # NOTE: In DCT source code, res50 has a bug here, kernel size = 3, but when copying weight,
            # kernel_size became 1. They probably mixed res18 and res50 by mistake
            temp_layer.weight.data = self.layer1[0].conv1.weight.data[:, :input_channels]
            self.layer1[0].conv1 = temp_layer

            # although conventional res18's BasicBlock doesn't have 'downsample',
            # we do need downsample here, otherwise,
            # the residual add operation will throw error
            downsample = self.layer1[0].downsample

            #didn't copy weights here, check if really needed
            # temp_layer.weight.data =
            if downsample is None:
                # res18
                out_ch = self.layer1[0].conv2.out_channels
                temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=3, stride=1, bias=False)
                downsample = nn.Sequential(
                    temp_layer,
                    self._norm_layer(64 * block.expansion),
                )
                self.layer1[0].downsample = downsample
            else:
                # res50
                out_ch = self.layer1[0].downsample[0].out_channels
                temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=1, stride=1, bias=False)
                temp_layer.weight.data = self.layer1[0].downsample[0].weight.data[:, :input_channels]
                self.layer1[0].downsample[0] = temp_layer

        else:
            out_ch = self.layer1[0].conv1.out_channels
            kernel_size = self.layer1[0].conv1.kernel_size
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=kernel_size, stride=1, bias=False)
            kaiming_init(temp_layer)
            self.layer1[0].conv1 = temp_layer

            out_ch = self.layer1[0].downsample[0].out_channels
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(temp_layer)
            self.layer1[0].downsample[0] = temp_layer

    def _forward_impl(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.maxpool(x) # without maxpool, then wptdownsample is needed in pre-processing

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DualResNet(nn.Module):
    def __init__(self, input_channels=64, pretrained=False):
        super(DualResNet, self).__init__()
        self.resnet152 = resnet152(pretrained=False)
        self.resnet = wpt_resnet_152(input_channels=input_channels, pretrained=pretrained)
        self.final_fc1 = nn.Linear(2000, 512)

        self.params = nn.ModuleDict({
            'base': nn.ModuleList([self.resnet152, self.final_fc1]),
            'wavelets': nn.ModuleList([self.resnet])})

    def forward(self, x, y):
        x = self.resnet152.conv1(x)
        x = self.resnet152.bn1(x)
        x = self.resnet152.relu(x)
        x = self.resnet152.maxpool(x)
        x = self.resnet152.layer1(x)
        x = self.resnet152.layer2(x)
        x = self.resnet152.layer3(x)
        x = self.resnet152.layer4(x)
        x = self.resnet152.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet152.fc(x)

        y = self.resnet.model.layer1(y)
        y = self.resnet.model.layer2(y)
        y = self.resnet.model.layer3(y)
        y = self.resnet.model.layer4(y)
        y = self.resnet.model.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.resnet.model.fc(y)

        h = torch.cat((x, y), 1)
        # h = x + y
        h = self.final_fc1(h)

        return h


def resnet_152(**kwargs):
    model = models.ResNet(models.resnet.Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


