from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.model_zoo as model_zoo
from visualDet3D.networks.utils.registry import BACKBONE_DICT


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = GhostModule(inplanes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, stride=1)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
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


class GhostResNet(nn.Module):
    planes = [64, 128, 256, 512]

    def __init__(self, block: Union[BasicBlock, Bottleneck],
                 layers: Tuple[int, ...],
                 num_stages: int = 4,
                 strides: Tuple[int, ...] = (1, 2, 2, 2),
                 dilations: Tuple[int, ...] = (1, 1, 1, 1),
                 out_indices: Tuple[int, ...] = (-1, 0, 1, 2, 3),
                 frozen_stages: int = -1,
                 norm_eval: bool = True,
                 ):
        '''
            # resnet arguments
            depth=34,
            pretrained=True,
            frozen_stages=-1,
            num_stages=3,
            out_indices=(0, 1, 2),
            norm_eval=True,
            dilations=(1, 1, 1),
        '''
        self.inplanes = 64
        super(GhostResNet, self).__init__()

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        assert max(out_indices) < num_stages

        # initial stage
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 初始 7x7 conv, stride=2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3x3 /2 maxpool

        # following stages: layer 1 2 3 4: [3, 4, 6, 3] for ghost_resnet34 (num_stages=3, actually [3, 4, 6] ? )
        for i in range(num_stages):
            setattr(self, f"layer{i + 1}", self._make_layer(block, self.planes[i], layers[i], stride=self.strides[i],
                                                            dilation=self.dilations[i]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # prior = 0.01

        self.train()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.inplanes, planes * block.expansion,
                #           kernel_size=1, stride=stride, bias=False),
                GhostModule(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(GhostResNet, self).train(mode)

        if mode:
            self.freeze_stages()
            if self.norm_eval:
                self.freeze_bn()

    def freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

            for param in self.bn1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):  # Will freeze both batchnorm and sync batchnorm
                layer.eval()

    def forward(self, img_batch):

        outs = []
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        if -1 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)
        for i in range(self.num_stages):
            layer = getattr(self, f"layer{i + 1}")
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


def ghost_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GhostResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def ghost_resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GhostResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def ghost_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GhostResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def ghost_resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GhostResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def ghost_resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GhostResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


# @BACKBONE_DICT.register_module
def ghost_resnet(depth, **kwargs):
    if depth == 18:
        model = ghost_resnet18(**kwargs)
    elif depth == 34:
        model = ghost_resnet34(**kwargs)
    elif depth == 50:
        model = ghost_resnet50(**kwargs)
    elif depth == 101:
        model = ghost_resnet101(**kwargs)
    elif depth == 152:
        model = ghost_resnet152(**kwargs)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    return model


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.in_planes = 64

        # Smooth layers
        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = x[0]
        c2 = x[1]
        c3 = x[2]
        # Top-down
        p3 = c3  # 256 channels
        p2 = self._upsample_add(self.latlayer3(p3), c2)  # 128 cnannels
        p1 = self._upsample_add(self.latlayer2(p2), c1)  # 64 cnannels
        # Smooth
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)
        return p1, p2, p3


if __name__ == '__main__':
    # model = ghost_resnet34(False).cuda()
    model = ghost_resnet(depth=34,  # resnet34
                         pretrained=True,
                         frozen_stages=-1,
                         num_stages=3,  # output stages (different scales)
                         out_indices=(0, 1, 2),
                         norm_eval=True,
                         dilations=(1, 1, 1),
                         ).cuda()
    model.eval()
    print(model)
    image = torch.rand(2, 3, 224, 224).cuda()

    output = model(image)
    for y in output:
        print(y.shape)
    #
    # fpn_model = FPN().cuda()
    # output2 = fpn_model(output)
    # for y in output2:
    #     print(y.shape)
