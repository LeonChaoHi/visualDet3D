from typing import Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.model_zoo as model_zoo
from visualDet3D.networks.utils.registry import BACKBONE_DICT
from thop import profile, clever_format     # package for calculating FLOPs and params

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount（通道数缩放系数）
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # t, c, n, s: 通道expand倍数, 输出通道数, 重复次数（类似一个stage, 包含几个block）, stride
        first_layers_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 64, 1, 2],
        ]
        stage1_setting = [
            [6, 32, 3, 1],
            [6, 64, 1, 1],
        ]
        stage2_setting = [
            [6, 48, 5, 2],
            [6, 128, 1, 1],
        ]
        stage3_setting = [
            [6, 96, 3, 2],
            [6, 256, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]  # init 层 stride 为 2
        # building inverted residual blocks
        for t, c, n, s in first_layers_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 在每一部分的第一个 block 调整 stride
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building stage1 stage2 stage3
        stage1_features = []
        # building inverted residual blocks
        for t, c, n, s in stage1_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 在每一部分的第一个 block 调整 stride
                stage1_features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.stage1 = nn.Sequential(*stage1_features)

        stage2_features = []
        # building inverted residual blocks
        for t, c, n, s in stage2_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 在每一部分的第一个 block 调整 stride
                stage2_features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.stage2 = nn.Sequential(*stage2_features)

        stage3_features = []
        # building inverted residual blocks
        for t, c, n, s in stage3_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1  # 在每一部分的第一个 block 调整 stride
                stage3_features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.stage3 = nn.Sequential(*stage3_features)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        outs = []
        x = self.features(x)
        # # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # x = self.classifier(x)
        x = self.stage1(x)
        outs.append(x)
        x = self.stage2(x)
        outs.append(x)
        x = self.stage3(x)
        outs.append(x)
        return outs

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = mobilenet_v2().cuda()
    model.eval()
    print(model)
    image = torch.rand(2, 3, 288, 1280).cuda()

    output = model(image)
    for y in output:
        print(y.shape)

    model_input = image
    macs, params = profile(model, (model_input,))
    macs, params = clever_format([macs, params], "%.3f")
    print('FLOPs:', macs)
    print('params:', params)
    print("FLOPs and params computation done.\n")

    print("start profiling inferencing time.")
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        model(model_input)
    print(prof)
    prof.export_chrome_trace('./mobilenet_v2_profile.json')


def records():
    # 0426 1st settings
    first_layers_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 32, 1, 2],
        [6, 64, 1, 1],
    ]
    stage1_setting = [
        [6, 32, 3, 1],
        [6, 64, 1, 1],
    ]
    stage2_setting = [
        [6, 64, 5, 2],
        [6, 128, 1, 1],
    ]
    stage3_setting = [
        [6, 128, 3, 2],
        [6, 256, 1, 1],
    ]