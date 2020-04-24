# adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py


import os
import collections
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET

from references.detection.transforms import Compose, RandomHorizontalFlip, ToTensor
from references.detection.utils import collate_fn


CLASSES = ['__background__', 'person' , 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            annotations: `root`/Annotations/*.xml
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)

        image_dir = '%s/JPEGImages/' % root
        annotation_dir = '%s/Annotations' % root
        splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)

        if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir) or not os.path.exists(splits_file):
            raise RuntimeError('Dataset not found or corrupted.')
        
        self.classes = CLASSES
        self.n_classes = len(self.classes)
        self.class2ind = {c: idx for c, idx in zip(CLASSES, range(self.n_classes))}

        file_names = np.loadtxt(splits_file, dtype=str)
        self.images = ['%s/%s.jpg' % (image_dir, x) for x in file_names]
        self.annotations = ['%s/%s.xml' % (annotation_dir, x) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        
        boxes, labels, iscrowd = [], [], []
        
        if not isinstance(target['annotation']['object'], list):
            target['annotation']['object'] = [target['annotation']['object']]
        for obj in target['annotation']['object']:
            xmin = int(obj['bndbox']['xmin']) - 1
            ymin = int(obj['bndbox']['ymin']) - 1
            xmax = int(obj['bndbox']['xmax']) - 1
            ymax = int(obj['bndbox']['ymax']) - 1
            boxes.append([xmin, ymin, xmax, ymax])    
            labels.append(self.class2ind[obj['name']])
            iscrowd.append(bool(int(obj['difficult'])))
        
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

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def get_transforms(is_train=False):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)


def load_data(root, batch_size, train_split='train', val_split='val', num_workers=0, max_samples=None):
    """
    `load train/val data loaders and class2ind (dict), n_classes (int)`

    Args:
        root (string): Root directory of the VOC Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            annotations: `root`/Annotations/*.xml
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
        batch_size: batch size for training
        train/val splits: `root`/ImageSets/Main/`image_set`.txt
        max_samples: maximum number of samples for train/val datasets. (Default: None)
            Can be set to a small number for faster training
    """
    train_dataset = VOCDetection(root, train_split, get_transforms(is_train=True))
    val_dataset = VOCDetection(root, val_split, get_transforms(is_train=False))

    class2ind = train_dataset.class2ind
    n_classes = train_dataset.n_classes

    if max_samples is not None:
        train_dataset = data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = data.Subset(val_dataset, np.arange(max_samples))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
                                   drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print('Number of Samples --> Train:%d\t Val:%d\t' % (len(train_dataset), len(val_dataset)))

    return train_loader, val_loader, class2ind, n_classes