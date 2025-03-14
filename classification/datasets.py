import os
import json
import yaml

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


class CustomDataset(Dataset):
    def __init__(self, labels, images_path, transform=None):
        self.root = images_path
        self.images_path = os.listdir(images_path)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.images_path)
    
    def _parse_label(self, path):
        for i,label in enumerate(self.labels):
            if '-'+label in path:
                return i

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.images_path[idx])
        label = self._parse_label(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    with open(args.data) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    if is_train:
        dataset = CustomDataset(config['labels'], config['train'], transform)
    else:
        dataset = CustomDataset(config['labels'], config['val'], transform)
    return dataset, config['labels']


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(transforms.ToPILImage())
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)