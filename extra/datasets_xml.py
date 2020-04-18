# adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
# https://github.com/jeremyfix/deeplearning-lectures/blob/master/LabsSolutions/01-pytorch-object-detection/data.py


import os
import tarfile
import collections
from PIL import Image
import torch
import torchvision
import xml.etree.ElementTree as ET

from references.detection.transforms import Compose, RandomHorizontalFlip, ToTensor


DATASET_YEAR_DICT = {
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    }
}


OBJECT_CLASSES = ['__background__', 'person' , 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

PART_CLASSES = []


class VOCDetection(torchvision.datasets.vision.VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, year='2010', image_set='train', transform=None, target_transform=None, transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + '.xml') for x in file_names]
        assert (len(self.images) == len(self.annotations))

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
            xmin = int(obj['bndbox']['xmin'])
            ymin = int(obj['bndbox']['ymin'])
            xmax = int(obj['bndbox']['xmax'])
            ymax = int(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])    
            labels.append(OBJECT_CLASSES.index(obj['name']))
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


def get_transform(is_train):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)
