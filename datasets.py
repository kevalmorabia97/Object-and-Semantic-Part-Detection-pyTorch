import json
import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset

from transforms import get_transforms


OBJECT_CLASSES = ['__background__', 'person' , 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class PascalPartVOCDetection(VisionDataset):
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
            class2ind_file: `root`/Classes/`class2ind_file`.txt
        image_set (string, optional): Select the image_set to use, e.g. train (default), trainval, val
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, required): A function/transform that takes in the target and transforms it.
        class2ind_file: file containing list of class names and class index that are to be considered from all annotations.
            Other object/part classes will be ignored.
            Default: `object_class2ind`.
            Note: `__background__` class should also be present.
        use_objects: if True (default: True), use object annotations.
        use_parts: if True (default: False), use part annotations that are present inside an object.
        return_separate_targets: if True, return img, obj_target, part_target instead of img, target (default: False)
            should be set True only for training JointDetector.
        part_class2ind_file: similar to `class2ind_file` but will have part classes (default: None).
            should be provided only if return_separate_targets=True otherwise should be provided as `class2ind_file`.
    """
    def __init__(self, root, image_set='train', transforms=None, transform=None, target_transform=None, class2ind_file='object_class2ind',
                 use_objects=True, use_parts=False, return_separate_targets=False, part_class2ind_file=None):
        super(PascalPartVOCDetection, self).__init__(root, transforms, transform, target_transform)

        image_dir = '%s/JPEGImages/' % root
        annotation_dir = '%s/Annotations_Part_json' % root
        splits_file = '%s/ImageSets/Main/%s.txt' % (root, image_set)
        class2ind_file = '%s/Classes/%s.txt' % (root, class2ind_file)

        if not os.path.isdir(image_dir) or not os.path.isdir(annotation_dir) or not os.path.exists(splits_file) or not os.path.exists(class2ind_file):
            raise RuntimeError('Dataset not found or corrupted.')
        if not use_objects and not use_parts:
            raise RuntimeError('Atleast 1 of objects and parts have to be used')
        self.use_objects = use_objects
        self.use_parts = use_parts
        self.return_separate_targets = return_separate_targets

        class2ind_list = np.loadtxt(class2ind_file, dtype=str) # shape [n_classes, 2]
        self.class2ind = {k: int(v) for k, v in class2ind_list}
        self.classes = set(self.class2ind.keys())
        self.n_classes = len(np.unique(list(self.class2ind.values())))

        if self.return_separate_targets:
            part_class2ind_file = '%s/Classes/%s.txt' % (root, part_class2ind_file)
            if not os.path.exists(part_class2ind_file):
                raise RuntimeError('For separate targets, class2ind_file is for objects and part_class2ind_file is for parts')
            class2ind_part_list = np.loadtxt(part_class2ind_file, dtype=str) # shape [n_classes, 2]
            self.part_class2ind = {k: int(v) for k, v in class2ind_part_list}
            self.part_classes = set(self.part_class2ind.keys())
            self.part_n_classes = len(np.unique(list(self.part_class2ind.values())))
        else:
            self.part_class2ind = self.class2ind
            self.part_classes = self.classes

        file_names = np.loadtxt(splits_file, dtype=str)
        # try x[0] if file_name is parsed incorrectly
        self.images = ['%s/%s.jpg' % (image_dir, x) for x in file_names]
        self.annotations = ['%s/%s.json' % (annotation_dir, x) for x in file_names]

        print('Image Set: %s,  Samples: %d, Objects: %s, Parts: %s, separate_targets: %s' % (image_set, len(self.images), use_objects, use_parts,
                                                                                             self.return_separate_targets))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary
                if return_separate_targets=True, returns (image, obj_target, part_target)
        """
        img = Image.open(self.images[index]).convert('RGB')

        if self.return_separate_targets:
            boxes, labels, iscrowd, part_boxes, part_labels, part_iscrowd = self.parse_json_annotation(self.annotations[index])
        else:
            boxes, labels, iscrowd = self.parse_json_annotation(self.annotations[index])
        
        if boxes == [] or (self.return_separate_targets and part_boxes == []):
            print('%s doesnt have any given objects/parts. Returning next image' % self.images[index])
            return self.__getitem__((index+1) % self.__len__())
        
        boxes = torch.Tensor(boxes)
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.LongTensor(labels)
        target['image_id'] = torch.tensor([index])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.BoolTensor(iscrowd)

        if self.return_separate_targets:
            part_boxes = torch.Tensor(part_boxes)
            part_target = {}
            part_target['boxes'] = part_boxes
            part_target['labels'] = torch.LongTensor(part_labels)
            part_target['image_id'] = torch.tensor([index])
            part_target['area'] = (part_boxes[:, 3] - part_boxes[:, 1]) * (part_boxes[:, 2] - part_boxes[:, 0])
            part_target['iscrowd'] = torch.BoolTensor(part_iscrowd)

            if self.transforms is not None:
                img, target, part_target = self.transforms(img, target, part_target)
            return img, target, part_target

        # if use_parts=True, parts will also be present in target
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def parse_json_annotation(self, file_path):
        target = json.load(open(file_path, 'r'))
        
        boxes, labels, iscrowd = [], [], []
        if self.return_separate_targets:
            part_boxes, part_labels, part_iscrowd = [], [], []
        else:
            part_boxes, part_labels, part_iscrowd = boxes, labels, iscrowd
        
        for obj in target['object']:
            if self.use_objects:
                if obj['name'] in self.classes: # ignore objects not in given list of classes
                    xmin = obj['bndbox']['xmin']
                    ymin = obj['bndbox']['ymin']
                    xmax = obj['bndbox']['xmax']
                    ymax = obj['bndbox']['ymax']
                    boxes.append([xmin, ymin, xmax, ymax]) 
                    labels.append(self.class2ind[obj['name']])
                    iscrowd.append(False)
            if self.use_parts:
                for part in obj['parts']:
                    if part['name'] in self.part_classes: # ignore parts not in given list of classes
                        xmin = part['bndbox']['xmin']
                        ymin = part['bndbox']['ymin']
                        xmax = part['bndbox']['xmax']
                        ymax = part['bndbox']['ymax']
                        part_boxes.append([xmin, ymin, xmax, ymax])
                        part_labels.append(self.part_class2ind[part['name']])
                        part_iscrowd.append(False)
        
        if self.return_separate_targets:
            return boxes, labels, iscrowd, part_boxes, part_labels, part_iscrowd
        return boxes, labels, iscrowd

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))


def load_data(root, batch_size, train_split='train', val_split='val', class2ind_file='object_class2ind', use_objects=True, use_parts=False,
              return_separate_targets=False, part_class2ind_file=None, num_workers=0, max_samples=None):
    """
    `load train/val data loaders and class2ind (dict), n_classes (int)`

    Args:
        root (string): Root directory of the Pascal Part Dataset. Must contain the fololowing dir structure:
            Images: `root`/JPEGImages/*.jpg
            Object and Part annotations: `root`/Annotations_Part_json/*.json [see `parse_Pascal_VOC_Part_Anno.py`]
            train/val splits: `root`/ImageSets/Main/`image_set`.txt
            class2ind_file: `root`/Classes/`class2ind_file`.txt
        batch_size: batch size for training
        train/val splits: `root`/ImageSets/Main/`image_set`.txt
        class2ind_file: file containing list of class names and class index that are to be considered from all annotations.
            Other object/part classes will be ignored.
            Default: `object_class2ind`.
            Note: `__background__` class should also be present.
        use_objects: if True (default=True), use object annotations
        use_parts: if True (default=False), use part annotations that are present inside an object
        return_separate_targets: if True, return img, obj_target, part_target instead of img, target (default: False)
            should be set True only for training JointDetector
        part_class2ind_file: similar to `class2ind_file` but will have part classes (default: None).
            should be provided only if return_separate_targets=True otherwise should be provided as `class2ind_file`.
        max_samples: maximum number of samples for train/val datasets. (Default: None)
            Can be set to a small number for faster training
    """
    train_dataset = PascalPartVOCDetection(root, train_split, get_transforms(is_train=True), class2ind_file=class2ind_file, use_objects=use_objects,
                                           use_parts=use_parts, return_separate_targets=return_separate_targets, part_class2ind_file=part_class2ind_file)
    val_dataset = PascalPartVOCDetection(root, val_split, get_transforms(is_train=False), class2ind_file=class2ind_file, use_objects=use_objects,
                                           use_parts=use_parts, return_separate_targets=return_separate_targets, part_class2ind_file=part_class2ind_file)

    class2ind = train_dataset.class2ind
    n_classes = train_dataset.n_classes

    if return_separate_targets:
        part_class2ind = train_dataset.part_class2ind
        part_n_classes = train_dataset.part_n_classes

    if max_samples is not None:
        train_dataset = data.Subset(train_dataset, np.arange(max_samples))
        val_dataset = data.Subset(val_dataset, np.arange(max_samples))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
                                   drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    if return_separate_targets:
        return train_loader, val_loader, class2ind, n_classes, part_class2ind, part_n_classes
    return train_loader, val_loader, class2ind, n_classes
