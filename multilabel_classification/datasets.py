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

class CustomDataset(Dataset):
    def __init__(self, labels_names, images_path, transform=None):
        self.root = images_path
        self.images_path = os.listdir(images_path)
        self.transform = transform
        self.labels_names = labels_names

    def __len__(self):
        return len(self.images_path)
    
    def _parse_label(self, path):
        labels = []
        for i,label in enumerate(self.labels_names):
            if '-'+label in path:
                labels.append(i)
        return labels

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images_path[idx])
        labels = self._parse_label(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image = image)['image']
        labels = self.encode_labels(labels)
        return image, labels
    
    def encode_labels(self, labels):
        encoded = np.zeros(len(self.labels_names))
        for label in labels:
            encoded[label] = 1
        return encoded        


def build_dataset(is_train, args):
    with open(args.data) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    transform = build_transform(is_train, args)
    groups = list(config['labels'].values())
    labels_names = [item for sublist in groups for item in sublist]
    if is_train:
        dataset = CustomDataset(labels_names, config['train'], transform)
    else:
        dataset = CustomDataset(labels_names, config['val'], transform)
        
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
        #t.append(transforms.CenterCrop(args.input_size))
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD,),
        ToTensorV2()
        ]
    return A.Compose(t)