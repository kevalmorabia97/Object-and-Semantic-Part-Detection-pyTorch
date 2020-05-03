import numpy as np
import random
import torch

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO


def convert_to_coco_api_obj_part(ds):
    """
    get obj/part_coco_dset (COCO) datasets from data in `ds` dataset (image, obj_target, part_target)
    Adapted from references/detection/coco_utils.py 
    """
    obj_coco, part_coco = COCO(), COCO()
    obj_dataset = {'images': [], 'categories': [], 'annotations': []}
    part_dataset = {'images': [], 'categories': [], 'annotations': []}
    obj_categories, part_categories = set(), set()
    obj_ann_id, part_ann_id = 1, 1 # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    for img_idx in range(len(ds)):
        # find better way to get target e.g. targets = ds.get_annotations(img_idx)
        img, obj_targets, part_targets = ds[img_idx]
        image_id = obj_targets['image_id'].item() # both targets have same image_id
        img_dict = {'id': image_id, 'height': img.shape[-2], 'width': img.shape[-1]}

        obj_dataset['images'].append(img_dict)
        obj_bboxes = obj_targets['boxes']
        obj_bboxes[:, 2:] -= obj_bboxes[:, :2]
        obj_bboxes = obj_bboxes.tolist()
        obj_labels = obj_targets['labels'].tolist()
        obj_areas = obj_targets['area'].tolist()
        obj_iscrowd = obj_targets['iscrowd'].tolist()

        part_dataset['images'].append(img_dict)
        part_bboxes = part_targets['boxes']
        part_bboxes[:, 2:] -= part_bboxes[:, :2]
        part_bboxes = part_bboxes.tolist()
        part_labels = part_targets['labels'].tolist()
        part_areas = part_targets['area'].tolist()
        part_iscrowd = part_targets['iscrowd'].tolist()
        
        if 'masks' in obj_targets:
            obj_masks = obj_targets['masks']
            obj_masks = obj_masks.permute(0, 2, 1).contiguous().permute(0, 2, 1) # make masks Fortran contiguous for coco_mask
            part_masks = part_targets['masks']
            part_masks = part_masks.permute(0, 2, 1).contiguous().permute(0, 2, 1) # make masks Fortran contiguous for coco_mask
        if 'keypoints' in obj_targets:
            obj_keypoints = obj_targets['keypoints']
            obj_keypoints = obj_keypoints.reshape(obj_keypoints.shape[0], -1).tolist()
            part_keypoints = part_targets['keypoints']
            part_keypoints = part_keypoints.reshape(part_keypoints.shape[0], -1).tolist()
        
        for i in range(len(obj_bboxes)):
            ann = {'id': obj_ann_id, 'image_id': image_id, 'bbox': obj_bboxes[i], 'category_id': obj_labels[i], 'area': obj_areas[i], 'iscrowd': obj_iscrowd[i]}
            obj_categories.add(obj_labels[i])
            if 'masks' in obj_targets:
                ann['segmentation'] = coco_mask.encode(obj_masks[i].numpy())
            if 'keypoints' in obj_targets:
                ann['keypoints'] = obj_keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in obj_keypoints[i][2::3])
            obj_dataset['annotations'].append(ann)
            obj_ann_id += 1
        
        for i in range(len(part_bboxes)):
            ann = {'id': part_ann_id, 'image_id': image_id, 'bbox': part_bboxes[i], 'category_id': part_labels[i], 'area': part_areas[i], 'iscrowd': part_iscrowd[i]}
            part_categories.add(part_labels[i])
            if 'masks' in part_targets:
                ann['segmentation'] = coco_mask.encode(part_masks[i].numpy())
            if 'keypoints' in part_targets:
                ann['keypoints'] = part_keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in part_keypoints[i][2::3])
            part_dataset['annotations'].append(ann)
            part_ann_id += 1
    
    obj_dataset['categories'] = [{'id': i} for i in sorted(obj_categories)]
    obj_coco.dataset = obj_dataset
    obj_coco.createIndex()

    part_dataset['categories'] = [{'id': i} for i in sorted(part_categories)]
    part_coco.dataset = part_dataset
    part_coco.createIndex()
    
    return obj_coco, part_coco


def get_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_intersection_area(box1, box2):
    """
    compute intersection area of box1 and box2 (both are 4 dim box coordinates in [x1, y1, x2, y2] format)
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    x_overlap = max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    y_overlap = max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    overlap_area = x_overlap * y_overlap
    
    return overlap_area


def is_box_inside(in_box, out_box):
    """
    check if in_box is inside out_box (both are 4 dim box coordinates in [x1, y1, x2, y2] format)
    """
    xmin_o, ymin_o, xmax_o, ymax_o = out_box
    xmin_i, ymin_i, xmax_i, ymax_i = in_box
    
    if (xmin_o > xmin_i) or (xmax_o < xmax_i) or (ymin_o > ymin_i) or (ymax_o < ymax_i):
        return False
    return True


def merge_targets(target1, target2):
    """
    merge 2 targets (dict) of the same image into single by merging their `boxes`, `labels`, `area`, `iscrowd`.
        All other keys in `target1` are copied in `merged_target`.
    return `merged_target` (dict), count of boxes in 1st target that can be use to split merged_target into 2 separate targets
    """
    merged_target = {}
    box_count_1 = len(target1['boxes'])

    for key in target1.keys():
        if key in ['boxes', 'labels', 'area', 'iscrowd']:
            merged_target[key] = torch.cat((target1[key], target2[key]))
        else:
            merged_target[key] = target1[key]

    return merged_target, box_count_1


def merge_targets_batch(targets1, targets2):
    """
    merge corresponding target from both `targets1`, `targets2` (lists of dict) using `merge_targets()`
    return `merged_targets`, `box_counts_1` that can be used to reverse the operation
    """
    merged_targets, box_counts_1 = [], []
    for t1, t2 in zip(targets1, targets2):
        merged_t, box_c_1 = merge_targets(t1, t2)
        merged_targets.append(merged_t); box_counts_1.append(box_c_1)
    
    return merged_targets, box_counts_1


def set_all_seeds(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def split_targets(merged_target, box_count_1):
    """
    split `merged_target` (dict) of an image into 2 by split their `boxes`, `labels`, `area`, `iscrowd`.
        For all the above dict keys, first `box_count_1` elements are put into `target1` and rest in `target2`
        All other keys in `merged_target` are copied in `target1` and `target2`.
    return `target1`, `target2` (dict)
    """
    target1, target2 = {}, {}

    for key in merged_target.keys():
        if key in ['boxes', 'labels', 'area', 'iscrowd']:
            target1[key] = merged_target[key][:box_count_1]
            target2[key] = merged_target[key][box_count_1:]
        else:
            target1[key] = target2[key] = merged_target[key]

    return target1, target2


def split_targets_batch(merged_targets, box_counts_1):
    """
    split each target from merged_targets (list of dict) into `targets1`, `targets2` (lists of dict) using `split_targets()`
    return `targets1`, `targets2` (list of dict)
    """
    targets1, targets2 = [], []
    for merged_t, box_c_1 in zip(merged_targets, box_counts_1):
        t1, t2 = split_targets(merged_t, box_c_1)
        targets1.append(t1); targets2.append(t2)
    
    return targets1, targets2


def visualize_bbox(img_path, target, plot_objects=True, plot_parts=True, out_img_path='bbox_viz.jpg'):
    """
    Required library: https://github.com/nalepae/bounding-box/
    """
    import cv2
    from bounding_box import bounding_box as bb

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    for obj in target['object']:
        if plot_objects:
            xmin = obj['bndbox']['xmin']
            ymin = obj['bndbox']['ymin']
            xmax = obj['bndbox']['xmax']
            ymax = obj['bndbox']['ymax']
            bb.add(img, xmin, ymin, xmax, ymax, obj['name'])
        if plot_parts:
            for part in obj['parts']:
                xmin = part['bndbox']['xmin']
                ymin = part['bndbox']['ymin']
                xmax = part['bndbox']['xmax']
                ymax = part['bndbox']['ymax']
                bb.add(img, xmin, ymin, xmax, ymax, part['name'])
    
    cv2.imwrite(out_img_path, img)
    cv2.imshow(target['filename'], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

