import pandas as pd
import numpy as np
import cv2
import os
import re
import argparse

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from  nextvit import nextvit_base
from datasets import build_dataloader
from torchvision.models.detection import FasterRCNN


def get_args_parser():
    parser = argparse.ArgumentParser('Next-ViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data', default='configs/data.yaml', type=str)
    # Model parameters
    parser.add_argument('--model', default='nextvit_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=736, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--flops', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.set_defaults(pin_mem=True)

    # distributed training parameters

    # test throught
    return parser

def main(args):
    train_dataloader, label_names = build_dataloader(is_train=True, args = args)
    val_dataloader, _ = build_dataloader(is_train=False, args = args)
    device = torch.device(args.device)
    backbone = nextvit_base(
        num_classes=5,
    )
    #model = FasterRCNN(backbone, num_classes=4)
    images, targets = next(iter(train_dataloader))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    print(targets)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Twins training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


    