import os
import json
import yaml
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Any
import csv
import random


class CustomDataset(Dataset):
    def __init__(self, images_path, data, transform=None):
        self.images_path = images_path
        self.images = data['images']
        self.annot = data['annotations']
        self.transform = transform

    def __len__(self):
        return len(self.images_path)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images[idx]['file_name'])
        image_id = self.images[idx]['id']
        labels = []
        bboxes = []
        for annot in self.annot:
            if annot['image_id']==image_id:
                bbox = annot['bbox']
                label = annot['category_id']
                bboxes.append(bbox)
                labels.append(label)
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image = image)['image']
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['labels'] = torch.stack(tuple(map(torch.tensor, zip(*sample['labels'])))).permute(1, 0)
        return image, target
    

def collate_fn(batch):
    return tuple(zip(*batch))

def build_dataloader(is_train, args):
    with open(args.data) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    transform = build_transform(is_train, args)
    images_path = config['images']
    labels_names = config['label_names']
    if is_train:
        with open(config['train_annot']) as f:
            train_annot = json.load(f)
        dataset = CustomDataset(
            images_path = images_path, 
            data = train_annot, 
            transform = transform
            )
        data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn
            )
    else:
        with open(config['val_annot']) as f:
            val_annot = json.load(f)
        dataset = CustomDataset(
            images_path = images_path, 
            data = val_annot, 
            transform = transform
            )
        data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn
            )
        
    return dataset, labels_names


def build_transform(is_train, args):
    with open(args.data) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    size_w = int((256 / 224) * config['width'])
    size_h = int((256 / 224) * config['height'])
    t = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            p=0.4,
            shift_limit=(-0.05, 0.05),
            scale_limit=(-0.2, 0.05),
            rotate_limit=(-20, 20),
            interpolation=1,
            border_mode=1,
            value=(0, 0, 0),
            mask_value=None
        ),
        A.RandomBrightnessContrast(p=0.4, brightness_limit=(-0.3, 0.1), contrast_limit=(-0.2, 0.5), brightness_by_max=True),
        A.HueSaturationValue(always_apply=False, p=0.4, hue_shift_limit=(-5, 9), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
        A.Resize(size_h, size_w),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD,),
        ToTensorV2()
        ]
    return A.Compose(t, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})