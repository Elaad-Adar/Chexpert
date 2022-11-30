import torch
from torchvision import models
import torch.nn as nn


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

            # TODO: didn't copy weights here, check if really needed
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

        # if self.use_gate:
        #     self.input_gate = GateModule(input_channels)
        #
        #     for name, m in self.named_modules():
        #         if 'inp_gate_l' in str(name):
        #             m.weight.data.normal_(0, 0.001)
        #             m.bias.data[::2].fill_(0.1)
        #             m.bias.data[1::2].fill_(2)
        #         elif 'inp_gate' in str(name):
        #             if isinstance(m, nn.Conv2d):
        #                 kaiming_init(m)
        #             elif isinstance(m, nn.BatchNorm2d):
        #                 constant_init(m, 1)

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


def wpt_resnet_152(input_channels, **kwargs):
    model = WPTResNet(input_channels, models.resnet.Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
