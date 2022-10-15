from .resnet import resnet101, resnet152, resnet18, resnet34, resnet50, ResNet, resnet
from .ghostnet import ghost_net
from .ghost_resnet import ghost_resnet101, ghost_resnet152, ghost_resnet18, ghost_resnet34, ghost_resnet50, GhostResNet, ghost_resnet
from .mobilenet_v2 import mobilenet_v2
from .shufflenet_v2 import shufflenet_v2_customized
from .dla import dlanet
from visualDet3D.networks.utils.registry import BACKBONE_DICT

def build_backbone(cfg):
    temp_cfg = cfg.copy()
    name = ""
    if 'name' in temp_cfg:
        name = temp_cfg.pop('name')
    else:
        name = 'resnet'

    return BACKBONE_DICT[name](**temp_cfg)
