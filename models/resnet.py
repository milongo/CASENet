# ResNet loading functions taken from Torchvision

import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from utils.losses import MultiLLFunction

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_no, stride=1, downsample=None):  # add dilation factor
        super(Bottleneck, self).__init__()
        if block_no < 5:
            dilation = 2
            padding = 2
        else:
            dilation = 4
            padding = 4

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SideOutput(nn.Module):

    def __init__(self, num_output, kernel_sz=None, stride=None):
        super(SideOutput, self).__init__()
        self.conv = nn.Conv2d(num_output, 20, 1, stride=1, padding=0, bias=True)

        if kernel_sz is not None:
            self.upsample = True
            self.upsampled = nn.ConvTranspose2d(20, 20, kernel_sz, stride=stride, padding=0, bias=False)
        else:
            self.upsample = False

    def forward(self, res):
        side_output = self.conv(res)
        if self.upsample:
            side_output = self.upsampled(side_output)

        return side_output


class Res5Output(nn.Module):

    def __init__(self, num_output=2048, kernel_sz=8, stride=8):
        super(Res5Output, self).__init__()
        self.conv = nn.Conv2d(num_output, 20, 1, stride=1, padding=0)
        self.upsampled = nn.ConvTranspose2d(20, 20, kernel_size=kernel_sz, stride=stride, padding=0)

    def forward(self, res):
        res = self.conv(res)
        res = self.upsampled(res)
        return res


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.res2 = self._make_layer(block, 64, layers[0], 2)  # res2
        self.res3 = self._make_layer(block, 256, layers[1], 3, stride=2)  # res3
        self.res4 = self._make_layer(block, 256, layers[2], 4, stride=2)  # res4
        self.res5 = self._make_layer(block, 512, layers[3], 5, stride=1)  # res5

        self.SideOutput1 = SideOutput(64)
        self.SideOutput2 = SideOutput(256, kernel_sz=4, stride=2)
        self.SideOutput3 = SideOutput(1024, kernel_sz=4, stride=4)
        self.Res5Output = Res5Output()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, block_no, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, block_no, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, block_no))

        return nn.Sequential(*layers)

    def _sliced_concat(self, res1, res2, res3, res5, num_classes):
        out_dim = num_classes * 4
        out_tensor = Variable(torch.FloatTensor(res1.size(0), out_dim, res1.size(2), res1.size(3))).cuda()
        for i in range(0, out_dim, 4):
            class_num = 0
            out_tensor[:, i, :, :] = res1[:, class_num, :, :]
            out_tensor[:, i + 1, :, :] = res2[:, class_num, :, :]
            out_tensor[:, i + 2, :, :] = res3[:, class_num, :, :]
            out_tensor[:, i + 3, :, :] = res5[:, class_num, :, :]
            class_num += 1

        return out_tensor

    def _fused_class(self, sliced_cat, groups):
        in_channels = sliced_cat.size(1)
        out_channels = sliced_cat.size(1)//groups
        conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups).cuda()
        out = conv(sliced_cat).cuda()
        return out

    def forward(self, x, labels):

        # res1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        side_1 = self.SideOutput1(x)

        # res2
        x = self.maxpool(x)
        x = self.res2(x)
        side_2 = self.SideOutput2(x)

        # res3
        x = self.res3(x)
        side_3 = self.SideOutput3(x)

        # res4
        x = self.res4(x)

        # res5
        x = self.res5(x)
        side_5 = self.Res5Output
        side_5 = side_5(x)

        # combine outputs and classify
        sliced_cat = self._sliced_concat(side_1, side_2, side_3, side_5, 20)
        acts = self._fused_class(sliced_cat, 4)
        preds = self.sigmoid(acts)

        return preds


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
