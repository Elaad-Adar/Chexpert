import torch
from torch.hub import load_state_dict_from_url
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
                                   bias=False)  # NOTE: In DCT source code, res50 has a bug here, kernel size = 3,
            # but when copying weight, kernel_size became 1. They probably mixed res18 and res50 by mistake
            temp_layer.weight.data = self.layer1[0].conv1.weight.data[:, :input_channels]
            self.layer1[0].conv1 = temp_layer

            # although conventional res18's BasicBlock doesn't have 'downsample',
            # we do need downsample here, otherwise,
            # the residual add operation will throw error
            downsample = self.layer1[0].downsample

            # didn't copy weights here, check if really needed
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



    def forward(self, x):
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


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def wpt_resnet_18(input_channels, pretrained=False, **kwargs):
    model = WPTResNet(input_channels, models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model


def wpt_resnet_34(input_channels, pretrained=False, **kwargs):
    model = WPTResNet(input_channels, models.resnet.BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34'],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model


def wpt_resnet_50(input_channels, pretrained=False, **kwargs):
    model = WPTResNet(input_channels, models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152'],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model


def wpt_resnet_152(input_channels, pretrained=False, **kwargs):
    model = WPTResNet(input_channels, models.resnet.Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    from torchsummary import summary
    import models.my_models

    summary(models.my_models.DualResNet(input_channels=32), [(3, 320, 320), (32, 32, 32)])

    # model = wpt_resnet_50(64)
    # summary(model, (64, 64, 64))

