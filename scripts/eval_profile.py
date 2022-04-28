
import importlib
import fire
import os
import copy
import torch

from _path_init import *
from visualDet3D.networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from visualDet3D.utils.utils import cfg_from_file
import os
import time
from tqdm import tqdm
from easydict import EasyDict
from typing import Sized, Sequence
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from thop import profile, clever_format     # package for calculating FLOPs and params
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from visualDet3D.evaluator.kitti.evaluate import evaluate
from visualDet3D.evaluator.kitti_depth_prediction.evaluate_depth import evaluate_depth
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection
from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt

"""
    created by liang
    used for model evaluation and profiling.
    FLOPs, inference time
"""

print('CUDA available: {}'.format(torch.cuda.is_available()))

@torch.no_grad()
def evaluate_kitti_obj(cfg: EasyDict,
                       model: nn.Module,
                       dataset_val: Sized,
                       writer: SummaryWriter,
                       epoch_num: int,
                       result_path_split='validation'
                       ):
    model.eval()
    # calculate FLOPs and params
    show_profile(0, dataset_val, model)
    return


def test_one(cfg, index, dataset, model, test_func, backprojector: BackProjection, projector: BBox3dProjector):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    original_height = data['original_shape'][0]
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]

    scores, bbox, obj_names = test_func(collated_data, model, None, cfg=cfg)
    bbox_2d = bbox[:, 0:4]
    if bbox.shape[1] > 4:  # run 3D
        bbox_3d_state = bbox[:, 4:]  # [cx,cy,z,w,h,l,alpha, bot, top]
        bbox_3d_state_3d = backprojector(bbox_3d_state, P2)  # [x, y, z, w,h ,l, alpha, bot, top]

        _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

        original_P = data['original_P']
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]

        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        # write_result_to_file(result_path, index, scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
    else:
        if "crop_top" in cfg.data.augmentation and cfg.data.augmentation.crop_top is not None:
            crop_top = cfg.data.augmentation.crop_top
        elif "crop_top_height" in cfg.data.augmentation and cfg.data.augmentation.crop_top_height is not None:
            if cfg.data.augmentation.crop_top_height >= original_height:
                crop_top = 0
            else:
                crop_top = original_height - cfg.data.augmentation.crop_top_height

        scale_2d = (original_height - crop_top) / height
        bbox_2d[:, 0:4] *= scale_2d
        bbox_2d[:, 1:4:2] += cfg.data.augmentation.crop_top
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        # write_result_to_file(result_path, index, scores, bbox_2d, obj_types=obj_names)


def show_profile(index, dataset, model):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    original_height = data['original_shape'][0]
    collated_data = dataset.collate_fn([data])
    left_images, right_images, P2, P3 = collated_data[0], collated_data[1], collated_data[2], collated_data[3]
    scores, bbox, obj_index = model([left_images.cuda().float().contiguous(), right_images.cuda().float().contiguous(),
                                     torch.tensor(P2).cuda().float(), torch.tensor(P3).cuda().float()])
    # model_input = np.concatenate([left_images, right_images], 0)
    # print(model_input.shape)
    model_input = [left_images.cuda().float().contiguous(), right_images.cuda().float().contiguous(),
                   torch.tensor(P2).cuda().float(), torch.tensor(P3).cuda().float()]
    macs, params = profile(model, (model_input,))
    macs, params = clever_format([macs, params], "%.3f")
    print('FLOPs:', macs)
    print('params:', params)
    print("FLOPs and params computation done.\n")

    print("start profiling inferencing time.")
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        model(model_input)
    print(prof)
    prof.export_chrome_trace('./resnet_profile.json')

def main(config:str="config/myyolo3d.py",
        gpu:int=0, 
        checkpoint_path:str=None,
        split_to_test:str='validation'):
    # Read Config
    cfg = cfg_from_file(config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)
    
    # Set up dataset and dataloader
    is_test_train = split_to_test == 'training'
    if split_to_test == 'training':
        dataset_name = cfg.data.train_dataset
    elif split_to_test == 'test':
        dataset_name = cfg.data.test_dataset
        cfg.is_running_test_set = True
    else:
        dataset_name = cfg.data.val_dataset
    dataset = DATASET_DICT[dataset_name](cfg, split_to_test)

    # Create the model
    detector = DETECTOR_DICT[cfg.detector.name](cfg.detector)
    detector = detector.cuda()

    assert checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
    new_dict = state_dict.copy()
    detector.load_state_dict(new_dict, strict=False)
    detector.eval()

    # Run evaluation
    evaluate_kitti_obj(cfg, detector, dataset, None, 0, result_path_split=split_to_test)
    print('finish')


if __name__ == '__main__':
    main("../config/myyolo3d.py", 0, "../runs/Stereo3D/model_results/0426_Stereo3D_49.pth") # TODO: path
