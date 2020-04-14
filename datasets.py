import json
import os
import numpy as np
from PIL import Image
import torch
import torchvision

from references.detection.transforms import Compose, RandomHorizontalFlip, ToTensor
from references.detection.utils import collate_fn


OBJECT_CLASSES = ['__background__', 'person' , 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

PART_CLASSES = []


class PascalPartVOCDetection(torchvision.datasets.vision.VisionDataset):
    """`Pascal Part VOC <http://host.robots.ox.ac.uk/pascal/VOC/>` Detection Dataset.
    Dataset links:
        PASCAL VOC 2010: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
        PASCAL-Parts annotations: http://roozbehm.info/pascal-parts/pascal-parts.html
    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
        image_set (string, optional): Select the image_set to use, e.g. train, trainval, val
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, required): A function/transform that takes in the target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, image_set='train', transform=None, target_transform=None, transforms=None):
        super(PascalPartVOCDetection, self).__init__(root, transforms, transform, target_transform)

        image_dir = '%s/JPEGImages/' % root
        annotation_dir = '%s/Annotations_Part_json' % root

        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)
        file_names = np.loadtxt(splits_file, dtype=str)
        self.images = ['%s/%s.jpg' % (image_dir, x) for x in file_names]
        self.annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = json.load(open(self.annotations[index], 'r'))
        
        boxes, labels, iscrowd = [], [], []
        for obj in target['object']:
            xmin = obj['bndbox']['xmin']
            ymin = obj['bndbox']['ymin']
            xmax = obj['bndbox']['xmax']
            ymax = obj['bndbox']['ymax']
            boxes.append([xmin, ymin, xmax, ymax])    
            labels.append(OBJECT_CLASSES.index(obj['name']))
            iscrowd.append(False)
        
        boxes = torch.Tensor(boxes)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.LongTensor(labels)
        target['image_id'] = torch.tensor([index])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.BoolTensor(iscrowd)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_transform(is_train):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)


def load_data(root, batch_size):
    """
    load train and val data loaders
    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
        batch_size: batch size for training
    """
    train_dataset = PascalPartVOCDetection(root, 'train', transforms=get_transform(is_train=True))
    val_dataset = PascalPartVOCDetection(root, 'val', transforms=get_transform(is_train=False))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader