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

## TODO: make classes more coarse grained
PART_CLASSES = ['__background__', 'backside', 'beak', 'bliplate', 'body', 'bwheel', 'cap',
                'cbackside_1', 'cbackside_2', 'cfrontside_1', 'cfrontside_2', 'cfrontside_3', 'cfrontside_4', 'cfrontside_5', 'cfrontside_6', 'cfrontside_7', 'cfrontside_9',
                'chainwheel', 'cleftside_1', 'cleftside_2', 'cleftside_3', 'cleftside_4', 'cleftside_5', 'cleftside_6', 'cleftside_7', 'cleftside_8', 'cleftside_9',
                'coach_1', 'coach_2', 'coach_3', 'coach_4', 'coach_5', 'coach_6', 'coach_7', 'coach_8', 'coach_9',
                'crightside_1', 'crightside_2', 'crightside_3', 'crightside_4', 'crightside_5', 'crightside_6', 'crightside_7', 'crightside_8',
                'croofside_1', 'croofside_2', 'croofside_3', 'croofside_4', 'croofside_5',
                'door_1', 'door_2', 'door_3', 'door_4', 'engine_1', 'engine_2', 'engine_3', 'engine_4', 'engine_5', 'engine_6',
                'fliplate', 'frontside', 'fwheel', 'hair', 'handlebar', 'hbackside', 'head', 'headlight_1',
                'headlight_2', 'headlight_3', 'headlight_4', 'headlight_5', 'headlight_6', 'headlight_7', 'headlight_8', 'hfrontside', 'hleftside', 'hrightside', 'hroofside',
                'lbho', 'lbleg', 'lblleg', 'lbpa', 'lbuleg', 'lear', 'lebrow', 'leftmirror', 'leftside', 'leye', 'lfho', 'lfleg', 'lflleg', 'lfoot', 'lfpa', 'lfuleg', 'lhand',
                'lhorn', 'llarm', 'lleg', 'llleg', 'luarm', 'luleg', 'lwing', 'mouth', 'muzzle', 'neck', 'nose', 'plant', 'pot',
                'rbho', 'rbleg', 'rblleg', 'rbpa', 'rbuleg', 'rear', 'rebrow', 'reye', 'rfho', 'rfleg', 'rflleg', 'rfoot', 'rfpa', 'rfuleg', 'rhand', 'rhorn', 'rightmirror', 'rightside', 'rlarm', 'rleg', 'rlleg',
                'roofside', 'ruarm', 'ruleg', 'rwing', 'saddle', 'screen', 'stern', 'tail', 'torso',
                'wheel_1', 'wheel_2', 'wheel_3', 'wheel_4', 'wheel_5', 'wheel_6', 'wheel_7', 'wheel_8',
                'window_1', 'window_10', 'window_11', 'window_12', 'window_13', 'window_14', 'window_15', 'window_16', 'window_17', 'window_18', 'window_19',
                'window_2', 'window_20', 'window_3', 'window_4', 'window_5', 'window_6', 'window_7', 'window_8', 'window_9']


class PascalPartVOCDetection(torchvision.datasets.vision.VisionDataset):
    """
    `Pascal Part VOC Detection Dataset`

    Dataset links:
        PASCAL VOC 2010: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
        PASCAL-Part annotations: http://roozbehm.info/pascal-parts/pascal-parts.html
    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
            classes_file (if provided): `root`/ImageSets/Main/`classes_file`.txt
        image_set (string, optional): Select the image_set to use, e.g. train (default), trainval, val
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, required): A function/transform that takes in the target and transforms it.
        classes_file: file containing list of class names that are to be considered from all annotations. Other object/part classes will be ignored.
            Default: None i.e. all object/part classes depending on values of `use_objects` and `use_parts`
            Note: `background` class should also be present
        use_objects: if True (default), use object annotations
        use_parts: if True (default), use part annotations that are present inside an object
    """
    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None, classes_file=None, use_objects=True, use_parts=True):
        super(PascalPartVOCDetection, self).__init__(root, transforms, transform, target_transform)

        image_dir = '%s/JPEGImages/' % root
        annotation_dir = '%s/Annotations_Part_json' % root
        splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)

        if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir) or not os.path.exists(splits_file):
            raise RuntimeError('Dataset not found or corrupted.')
        if not use_objects and not use_parts:
            raise RuntimeError('Atleast 1 of objects and parts have to be used')
        self.use_objects = use_objects
        self.use_parts = use_parts

        if classes_file is None:
            classes = []
            if self.use_objects:
                classes += OBJECT_CLASSES
            if self.use_parts:
                classes += PART_CLASSES
            classes = sorted(list(set(classes)))
        else:
            classes = list(np.loadtxt('%s/ImageSets/Main/%s.txt' % (root, classes_file), dtype=str))
        self.classes = classes

        file_names = np.loadtxt(splits_file, dtype=str)
        self.images = ['%s/%s.jpg' % (image_dir, x) for x in file_names]
        self.annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]

        print('Use Objects: %s, Use Parts: %s, No. of Classes: %d for %s image set' % (use_objects, use_parts, len(self.classes), image_set))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary
        """
        img = Image.open(self.images[index]).convert('RGB')
        boxes, labels, iscrowd = self.parse_json_annotation(self.annotations[index])
        
        if boxes == []:
            print('%s doesnt have any given objects/parts. Returning next image' % self.images[index])
            return self.__getitem__((index+1) % self.__len__())
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

    def parse_json_annotation(self, file_path):
        target = json.load(open(file_path, 'r'))
        
        boxes, labels, iscrowd = [], [], []
        for obj in target['object']:
            if self.use_objects:
                if obj['name'] in self.classes: # ignore objects not in given list of classes
                    xmin = obj['bndbox']['xmin']
                    ymin = obj['bndbox']['ymin']
                    xmax = obj['bndbox']['xmax']
                    ymax = obj['bndbox']['ymax']
                    boxes.append([xmin, ymin, xmax, ymax]) 
                    labels.append(self.classes.index(obj['name']))
                    iscrowd.append(False)
            if self.use_parts:
                for part in obj['parts']:
                    if part['name'] in self.classes: # ignore parts not in given list of classes
                        xmin = part['bndbox']['xmin']
                        ymin = part['bndbox']['ymin']
                        xmax = part['bndbox']['xmax']
                        ymax = part['bndbox']['ymax']
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.classes.index(part['name']))
                        iscrowd.append(False)
        
        return boxes, labels, iscrowd

    def __len__(self):
        return len(self.images)


def get_transforms(is_train=False):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)


def load_data(root, batch_size, train_split='train', val_split='val', classes_file=None, use_objects=True, use_parts=True, num_workers=0, max_samples=None):
    """
    `load train and val data loaders`

    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
            classes_file (if provided): `root`/ImageSets/Main/`classes_file`.txt
        batch_size: batch size for training
        train/val splits: `root`/ImageSets/Main/`image_set`.txt
        classes_file: file containing list of class names that are to be considered from all annotations. Other object/part classes will be ignored.
            Default: None i.e. all object/part classes depending on values of `use_objects` and `use_parts`
            Note: `background` class should also be present
        use_objects: if True (default), use object annotations
        use_parts: if True (default), use part annotations that are present inside an object
        max_samples: maximum number of samples for train/val datasets. (Default: None)
            Can be set to a small number for faster training
    """
    train_dataset = PascalPartVOCDetection(root, train_split, get_transforms(is_train=True), classes_file=classes_file, use_objects=use_objects, use_parts=use_parts)
    val_dataset = PascalPartVOCDetection(root, val_split, get_transforms(is_train=False), classes_file=classes_file, use_objects=use_objects, use_parts=use_parts)

    if max_samples is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(max_samples))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    print('Number of Samples --> Train:%d\t Val:%d\t' %(len(train_dataset), len(val_dataset)))

    return train_loader, val_loader