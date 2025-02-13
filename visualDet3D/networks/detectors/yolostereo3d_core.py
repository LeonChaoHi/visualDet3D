import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.lib.ghost_module import ResGhostModule, GhostModule
from visualDet3D.networks.lib.PSM_cost_volume import PSMCosineModule, CostVolume
from visualDet3D.networks.backbones import resnet, ghost_net, ghost_resnet, mobilenet_v2, shufflenet_v2_customized
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.look_ground import LookGround

'''
    Stereo 3d core
    including backbone and neck
    backbone: resnet
    neck: PSVolume, 
'''


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


class HourGlass(nn.Module):
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(HourGlass, self).__init__()
        self.depth_channel_4 = depth_channel_4  # 24
        self.depth_channel_8 = depth_channel_8  # 24
        self.depth_channel_16 = depth_channel_16  # 96

        input_features = depth_channel_4 # 24
        self.four_ghost = ResGhostModule(input_features, 3 * input_features, 3, ratio=3)
        self.four_to_eight = nn.Sequential(
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_ghost = ResGhostModule(input_features, 3 * input_features, 3, ratio=3)
        self.eight_to_sixteen = nn.Sequential(
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        self.output_channel_num = 3 * input_features #1152

        # TODO: depth output 修改
        self.HG1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),                   # s = 8, c = 1152
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),   # c = 576
            nn.BatchNorm2d(int(self.output_channel_num/2)),                                    # c = 576, add after this
        )
        self.HG2 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),   # s = 4, c = 576
            nn.Conv2d(int(self.output_channel_num/2) + 288, int(self.output_channel_num/4), 3, padding=1),    # c = 288
            nn.BatchNorm2d(int(self.output_channel_num/4)),                     # c = 288, add after this
        )
        self.HG3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4) + 72, 96, 1),
        )

    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4 = self.four_ghost(psv_volume_4)                       # new : s = 4, c = 72
        psv_4_8 = self.four_to_eight(psv_4)                         # s = 8, c = 72
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)    # s = 8, c = 96
        psv_8 = self.eight_ghost(psv_volume_8)                      # new : s = 8, c = 288
        psv_8_16 = self.eight_to_sixteen(psv_8)                     # s = 16, c = 288
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1) # s = 16, c = 384
        psv_16 = self.depth_reason(psv_volume_16)                   # s = 16, c = 1152
        if self.training:
            feat_8 = self.HG1(psv_16)
            feat_8 = torch.cat([feat_8, psv_8], dim=1)
            feat_4 = self.HG2(feat_8)
            feat_4 = torch.cat([feat_4, psv_4], dim=1)
            return psv_16, self.HG3(feat_4)
        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])


class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4  = depth_channel_4 # 24
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_4 # 24
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        self.output_channel_num = 3 * input_features #1152

        # TODO: depth output 修改
        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),   # s = 8, c = 1152
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),   # c = 576
            nn.BatchNorm2d(int(self.output_channel_num/2)), # c = 576
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),   # s = 4, c = 576
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),    # c = 288
            nn.BatchNorm2d(int(self.output_channel_num/4)), # c = 288
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )


    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)                  # s = 8, c = 72
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)    # add : s = 8, c = 96
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)              # s = 16, c = 288
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1) # s = 16, c = 384
        psv_16 = self.depth_reason(psv_volume_16)                   # s = 16, c = 1152
        if self.training:
            return psv_16, self.depth_output(psv_16)
        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])

class StereoMerging(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=96, input_features=base_features)
        PSV_depth_0 = self.cost_volume_0.depth_channel

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=base_features * 2)
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=base_features * 4, PSM_features=8)
        PSV_depth_2 = self.cost_volume_2.output_channel

        # TODO : new merging structure
        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        # self.depth_reasoning = HourGlass(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        # PSV_features 和 depth_output 均在 stereo_merging 中计算.
        # 此处不分 test 或 train, 均会计算 depth_output 部分
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408  叠加 left特征图 和 PSM block
        return features, depth_output

class YoloStereo3DCore(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore, self).__init__()
        # TODO: backbone defined here. ** stand for trans arguments by dict
        ''' original backbone '''
        self.backbone = resnet(**backbone_arguments) # resnet_34
        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        ''' ghost net '''
        # self.backbone = ghost_net()
        # base_features = 256 if backbone_arguments['depth'] > 34 else 64
        ''' ghost resnet '''
        # self.backbone = ghost_resnet(**backbone_arguments)
        # base_features = 256 if backbone_arguments['depth'] > 34 else 64
        ''' mobilenet v2 '''
        # self.backbone = mobilenet_v2()
        # base_features = 256 if backbone_arguments['depth'] > 34 else 64
        ''' shufflenet v2 '''
        # self.backbone = shufflenet_v2_customized()
        # base_features = 256 if backbone_arguments['depth'] > 34 else 64   # TODO: base features: output channels == 64
        ''' FPN module '''
        self.fpn = FPN()
        ''' neck '''
        self.neck = StereoMerging(base_features)


    def forward(self, images):

        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        images = torch.cat([left_images, right_images], dim=0)      # concat left_images and right_images in batch dim

        features = self.backbone(images)
        features = self.fpn(features)   # TODO: FPN

        left_features  = [feature[0:batch_size] for feature in features]    # split features from left and right images
        right_features = [feature[batch_size:]  for feature in features]

        features, depth_output = self.neck(left_features, right_features)

        output_dict = dict(features=features, depth_output=depth_output)
        return output_dict
