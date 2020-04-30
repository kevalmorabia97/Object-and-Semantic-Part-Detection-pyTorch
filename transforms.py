import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, target2=None):
        for t in self.transforms:
            if target2 is not None:
                image, target, target2 = t(image, target, target2)
            else:
                image, target = t(image, target)
        
        if target2 is not None:
            return image, target, target2
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, target2=None):
        image = F.normalize(image, self.mean, self.std)
        
        if target2 is not None:
            return image, target, target2
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob
    
    def _flip_target(self, target, width):
        bbox = target["boxes"]
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = _flip_coco_person_keypoints(keypoints, width)
            target["keypoints"] = keypoints
        
        return target

    def __call__(self, image, target, target2=None):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            target = self._flip_target(target, width)
            if target2 is not None:
                target2 = self._flip_target(target2, width)
        
        if target2 is not None:
            return image, target, target2
        return image, target


class ToTensor(object):
    def __call__(self, image, target, target2=None):
        image = F.to_tensor(image)
        
        if target2 is not None:
            return image, target, target2
        return image, target


def get_transforms(is_train=False):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    return Compose(transforms)