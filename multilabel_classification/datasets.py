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

class MultilabelDataset(Dataset):
    def __init__(self,
                 nc,
                 annotation_path: str,
                 images_path: str,
                 labels: Dict[str, List[str]],
                 balance_classess: bool = True,
                 transform=None,
                 image_limit = None,
                 batch_size: int = None,
                 class_weights=None,
                 augs_type = "albumentations"
                 ):
        super().__init__()
        
        self._nc = nc
        self._transform = transform
        self._batch_size = batch_size
        self._balance_classess = balance_classess

        # initialize the arrays to store the ground truth labels and paths to the images
        self.image_names: List[str] = self._get_image_names(annotation_path, images_path, limit=image_limit)
        self.labels: Dict[str, List[str]] = labels

        # read the annotations from the CSV file
        self.image_annotations: Dict[str, List[str]] = self._get_image_annotations(
            annotation_path, limit=image_limit
        )
        self._augs_type = augs_type

        if balance_classess:
            self._resample_classess(class_weights)

    def __len__(self):
        if self._batch_size is not None:
            return math.ceil(len(self.image_names) / self._batch_size)
        return len(self.image_names)

    def __getitem__(self, batch_idx):

        if batch_idx >= len(self):
            raise StopIteration

        # take the data sample by its index
        if self._batch_size is not None:
            img = list()
            labels = {label_name: list() for label_name in self.labels}
            for img_idx in range(batch_idx * self._batch_size, min((batch_idx + 1) * self._batch_size, len(self.image_names))):
                img.append(self._get_image(self.image_names[img_idx]))
                for label_name in self.labels:
                    labels[label_name].append(self.image_annotations[label_name][img_idx])
        else:
            img = self._get_image(self.image_names[batch_idx])
            labels = {
                label_name: self.image_annotations[label_name][batch_idx]
                for label_name in self.labels
            }
            
        labels = self.encode_labels(labels)
            
        return img,labels
    
    def encode_labels(self, labels):
        encoded = np.zeros(self._nc)
        for i,label in enumerate(list(labels.values())):
            encoded[i*3+label] = 1
        return encoded     

    def get_labels_number_by_name(self) -> Dict[str, int]:
        return {
            attribute_name: len(attribute_labels)
            for attribute_name, attribute_labels in self.labels.items()
        }

    def _resample_classess(self, class_weights=None, dataset_scaling: int = 3):
        
        label_counter = {group_name: {label_name: 0 for label_name in self.labels[group_name]} for group_name in self.labels}
        for label_group_name in self.image_annotations:
            for label_index in self.image_annotations[label_group_name]:
                label_counter[label_group_name][self.labels[label_group_name][label_index]] += 1

        if class_weights is None:
            not_in_hardhat_count = label_counter['hardhat']['not_in_hardhat']
            in_hardhat_count = label_counter['hardhat']['in_hardhat']
            hardhat_unrecognized_count = label_counter['hardhat']['hardhat_unrecognized']

            not_in_vest_count = label_counter['vest']['not_in_vest']
            in_vest_count = label_counter['vest']['in_vest']
            vest_unrecognized_count = label_counter['vest']['vest_unrecognized']

            not_in_harness_count = label_counter['harness']['not_in_harness']
            in_harness_count = label_counter['harness']['in_harness']
            harness_unrecognized_count = label_counter['harness']['harness_unrecognized']
            
            person_in_bucket_count = label_counter['person_in_bucket']['person_in_bucket']
            person_not_in_bucket_count = label_counter['person_in_bucket']['person_not_in_bucket']

            harness_total = not_in_harness_count + in_harness_count + hardhat_unrecognized_count
            hardhat_total = not_in_hardhat_count + in_hardhat_count + hardhat_unrecognized_count
            vest_total = not_in_vest_count + in_vest_count + hardhat_unrecognized_count
            person_in_bucket_total = person_in_bucket_count + person_not_in_bucket_count 

            hardhat_infraction_probability = not_in_hardhat_count / hardhat_total
            harness_infraction_probability = not_in_harness_count  / harness_total
            vest_infraction_probability = not_in_vest_count / vest_total

            in_hardhat_probability = in_hardhat_count / hardhat_total
            in_harness_probability = in_harness_count / harness_total
            in_vest_probability = in_vest_count / vest_total

            hardhat_unrecognized_probability = hardhat_unrecognized_count / hardhat_total
            harness_unrecognized_probability = harness_unrecognized_count / harness_total
            vest_unrecognized_probability = vest_unrecognized_count / vest_total
            
            person_in_bucket_probability = person_in_bucket_count / person_in_bucket_total
            person_not_in_bucket_probability = person_not_in_bucket_count  / person_in_bucket_total

            class_weights = {
                'hardhat': {
                    'in_hardhat': hardhat_infraction_probability,
                    'not_in_hardhat': in_hardhat_probability,
                    'hardhat_unrecognized': hardhat_unrecognized_probability,
                },
                'harness': {
                    'in_harness': harness_infraction_probability,
                    'not_in_harness': in_harness_probability,
                    'harness_unrecognized': harness_unrecognized_probability,
                },
                'vest': {
                    'in_vest': vest_infraction_probability,
                    'not_in_vest': in_vest_probability,
                    'vest_unrecognized': vest_unrecognized_probability,
                },
                'person_in_bucket':{
                    'person_in_bucket':person_in_bucket_probability,
                    'person_not_in_bucket':person_not_in_bucket_probability
                }
            }
        else:
            print('Got class weights from config', class_weights)

        selected_image_names = list()
        selected_annotations = {class_group_name: list() for class_group_name in self.labels}

        ann_limit = len(self.image_names) * dataset_scaling

        collect_annotations = True

        counter = 0
        while collect_annotations:
            counter += 1
            for i, image_name in enumerate(self.image_names):
                ann_weights = list()
                for class_group_name in self.labels:
                    label = self.labels[class_group_name][self.image_annotations[class_group_name][i]]
                    ann_weights.append(class_weights[class_group_name][label])
                mean_ann_weights = ann_weights[0] * ann_weights[1] * ann_weights[2]
                if random.uniform(0, 1) < mean_ann_weights:
                    selected_image_names.append(image_name)
                    for class_group_name in self.labels:
                        selected_annotations[class_group_name].append(self.image_annotations[class_group_name][i])
                if len(selected_image_names) >= ann_limit:
                    collect_annotations = False
                    break

        label_counter = {group_name: {label_name: 0 for label_name in self.labels[group_name]} for group_name in self.labels}
        for label_group_name in self.image_annotations:
            for label_index in self.image_annotations[label_group_name]:
                label_counter[label_group_name][self.labels[label_group_name][label_index]] += 1
        print('Classess before resampling:', label_counter)

        self.image_names = selected_image_names
        self.image_annotations = selected_annotations

        label_counter = {group_name: {label_name: 0 for label_name in self.labels[group_name]} for group_name in self.labels}
        for label_group_name in self.image_annotations:
            for label_index in self.image_annotations[label_group_name]:
                label_counter[label_group_name][self.labels[label_group_name][label_index]] += 1
        print('Classess after resampling:', label_counter)

    def _get_image_annotations(self, annotation_path: str, limit: int) -> Dict[str, List[str]]:
        image_annotations = {label_name: list() for label_name in self.labels}
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
            if limit is not None:
                rows = rows[:limit]
            for row in rows:
                for label_name in self.labels:
                    image_annotations[label_name].append(
                        self.labels[label_name].index(row[label_name])
                    )
        return image_annotations

    def _get_image(self, img_path: str,) -> np.ndarray:
        if self._transform is not None:
            if self._augs_type == "torchvision":
                img = self._transform(Image.open(img_path))
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._transform(image=img)['image']
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _get_image_names(annotation_path: str, images_path: str, limit: int) -> List[str]:
        image_names = list()
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
            if limit is not None:
                rows = rows[:limit]
            for row in rows:
                image_names.append(os.path.join(images_path, row["image_path"]))
        return image_names


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
        dataset = MultilabelDataset(
            nc = config['nc'],
            annotation_path = config['train_csv'], 
            images_path = config['images'], 
            transform = transform, 
            labels = config['labels'],
            balance_classess=args.balance_classess)
    else:
        dataset = MultilabelDataset(
            nc = config['nc'],
            annotation_path = config['val_csv'], 
            images_path = config['images'], 
            transform = transform, 
            labels = config['labels'],
            balance_classess=False)
        
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