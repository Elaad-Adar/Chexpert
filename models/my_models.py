from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from torchvision.models import resnet152
from models.wpt_net import wpt_resnet_50


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


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        # Essentially the 1x1 conv performs the down-sampling from num_input_features to num_output_features.
        self.add_module('conv', nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: #(List[Tensor]) -> Tensor

        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple denseblocks based on config
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


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
        self.resnet = wpt_resnet_50(input_channels=input_channels, pretrained=pretrained)
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


def wpt_resnet_152(input_channels, **kwargs):
    model = WPTResNet(input_channels, models.resnet.Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def _densenet121(arch, growth_rate, block_config, num_init_features, pretrained, progress,
                 **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet121('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                        **kwargs)
