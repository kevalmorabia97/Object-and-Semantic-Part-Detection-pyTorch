"""
Code to train the Joint Object and Part Detector
"""

import argparse
import numpy as np
import os
import torch

from datasets import load_data
from models import JointDetector
from train_joint import train_one_epoch, evaluate
from utils import set_all_seeds


########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-dir', '--data_dir', type=str, default='data/VOCdevkit/VOC2010/')
parser.add_argument('-tr', '--train_split', type=str, default='animals_train')
parser.add_argument('-val', '--val_split', type=str, default='animals_val')
parser.add_argument('-ocf', '--obj_class2ind_file', type=str, default='animals_object_class2ind')
parser.add_argument('-pcf', '--part_class2ind_file', type=str, default='animals_part_mergedclass2ind')
parser.add_argument('-e', '--n_epochs', type=int, default=30)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-bs', '--batch_size', type=int, default=1)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
parser.add_argument('-ft', '--fusion_thresh', type=float, default=0.9)
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-ms', '--max_samples', type=int, default=-1)
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(123)

########## Parameters ##########
DATA_DIR = args.data_dir
TRAIN_SPLIT = args.train_split
VAL_SPLIT = args.val_split
OBJ_CLASS2IND_FILE = args.obj_class2ind_file
PART_CLASS2IND_FILE = args.part_class2ind_file
N_EPOCHS = args.n_epochs
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None
USE_OBJECTS = True
USE_PARTS = True
RETURN_SEPARATE_TARGETS = True

model_save_path = 'saved_model_joint_%s.pth' % (TRAIN_SPLIT)

########## Hyperparameters ##########
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay
FUSION_THRESH = args.fusion_thresh

########## Data Loaders ##########
train_loader, val_loader, obj_class2ind, obj_n_classes, part_class2ind, part_n_classes = load_data(DATA_DIR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
    OBJ_CLASS2IND_FILE, USE_OBJECTS, USE_PARTS, RETURN_SEPARATE_TARGETS, PART_CLASS2IND_FILE, NUM_WORKERS, MAX_SAMPLES)

########## Create Model ##########
model = JointDetector(obj_n_classes, part_n_classes, FUSION_THRESH).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

########## Restore Saved Model If Exists ##########
start_epoch = 0
best_val_mAP = {'OBJECT': 0., 'PART': 0.}
if os.path.exists(model_save_path):
    print('Restoring trained model from %s' % model_save_path)
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_val_mAP = checkpoint['mAP']
    start_epoch = checkpoint['epoch']+1
    N_EPOCHS = start_epoch + N_EPOCHS

########## Train Model ##########
for epoch in range(start_epoch, N_EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=250)
    _, _, obj_det_stats, part_det_stats = evaluate(model, val_loader, device=device, print_freq=500, header='Val:')
    lr_scheduler.step()

    val_obj_det_mAP = obj_det_stats['bbox'][1] # AP @ IoU=0.5
    val_part_det_mAP = part_det_stats['bbox'][1] # AP @ IoU=0.5
    if val_obj_det_mAP > best_val_mAP['OBJECT']: # Save model which performs best for object detection
        best_val_mAP = {'OBJECT': val_obj_det_mAP, 'PART': val_part_det_mAP}
        checkpoint = {'model': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
                      'epoch': epoch, 'mAP': best_val_mAP}
        torch.save(checkpoint, model_save_path)
    
    print('-'*100)

print('Best val_mAP OBJECT: %.2f%%, PART: %.2f%%' % (100*best_val_mAP['OBJECT'], 100*best_val_mAP['PART']))
