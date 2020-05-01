# Adapted from references/detection/engine.py for joint training of obj and part

import math
import sys
import time
import torch
import torchvision.models.detection

from references.detection.coco_eval import CocoEvaluator
from references.detection.utils import MetricLogger, SmoothedValue, reduce_dict, warmup_lr_scheduler
from utils import convert_to_coco_api_obj_part


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1000):
    """
    Train model (JointDetector) for 1 epoch from data in data_loader (images, obj_targets, part_targets) 
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, obj_targets, part_targets in metric_logger.log_every(data_loader, print_freq, header, device):
        images = list(image.to(device) for image in images)
        obj_targets = [{k: v.to(device) for k, v in t.items()} for t in obj_targets]
        part_targets = [{k: v.to(device) for k, v in t.items()} for t in part_targets]

        loss_dict = model(images, obj_targets, part_targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device, print_freq=1000, header='Test:'):
    """
    Evaluate model (JointDetector) from data in data_loader (images, obj_targets, part_targets)
    Return obj/part_coco_evaluator, obj/part_stats
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')

    obj_coco, part_coco = convert_to_coco_api_obj_part(data_loader.dataset)
    iou_types = ['bbox']
    obj_coco_evaluator = CocoEvaluator(obj_coco, iou_types)
    part_coco_evaluator = CocoEvaluator(part_coco, iou_types)

    for images, obj_targets, part_targets in metric_logger.log_every(data_loader, print_freq, header, device):
        images = list(image.to(device) for image in images)
        obj_targets = [{k: v.to(device) for k, v in t.items()} for t in obj_targets]
        part_targets = [{k: v.to(device) for k, v in t.items()} for t in part_targets]

        torch.cuda.synchronize()
        model_time = time.time()
        obj_detections, part_detections = model(images, obj_targets, part_targets)
        
        obj_detections = [{k: v.to(cpu_device) for k, v in t.items()} for t in obj_detections]
        part_detections = [{k: v.to(cpu_device) for k, v in t.items()} for t in part_detections]
        model_time = time.time() - model_time

        obj_res = {target['image_id'].item(): output for target, output in zip(obj_targets, obj_detections)}
        part_res = {target['image_id'].item(): output for target, output in zip(part_targets, part_detections)}

        evaluator_time = time.time()
        obj_coco_evaluator.update(obj_res)
        part_coco_evaluator.update(part_res)
        evaluator_time = time.time() - evaluator_time
        
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    obj_coco_evaluator.synchronize_between_processes()
    part_coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    print('\nObject Detection Results:')
    obj_coco_evaluator.accumulate()
    obj_stats = obj_coco_evaluator.summarize()
    
    print('\nPart Detection Results:')
    part_coco_evaluator.accumulate()
    part_stats = part_coco_evaluator.summarize()
    
    torch.set_num_threads(n_threads)
    return obj_coco_evaluator, part_coco_evaluator, obj_stats, part_stats
