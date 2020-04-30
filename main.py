import argparse
import numpy as np
import os
import torch

from datasets import load_data
# from extra.datasets_xml import load_data
from models import get_FasterRCNN_model
from references.detection.engine import train_one_epoch, evaluate
from utils import set_all_seeds


########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-dir', '--data_dir', type=str, default='data/VOCdevkit/VOC2010/')
parser.add_argument('-tr', '--train_split', type=str, default='train')
parser.add_argument('-val', '--val_split', type=str, default='val')
parser.add_argument('-cf', '--class2ind_file', type=str, default='object_class2ind')
parser.add_argument('-e', '--n_epochs', type=int, default=30)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-bs', '--batch_size', type=int, default=1)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6)
parser.add_argument('--use_objects', dest='use_objects', action='store_true')
parser.add_argument('--use_parts', dest='use_parts', action='store_true')
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-ms', '--max_samples', type=int, default=-1)
args = parser.parse_args()

device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
set_all_seeds(123)

########## Parameters ##########
DATA_DIR = args.data_dir
TRAIN_SPLIT = args.train_split
VAL_SPLIT = args.val_split
CLASS2IND_FILE = args.class2ind_file
N_EPOCHS = args.n_epochs
USE_OBJECTS = bool(args.use_objects)
USE_PARTS = bool(args.use_parts)
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None

if USE_OBJECTS and USE_PARTS:
    print('[WARNING]: If you are doing Object and Part Detection, make sure you are using the class2ind file that has both classes')

model_save_path = 'saved_model_%s.pth' % (TRAIN_SPLIT)

########## Hyperparameters ##########
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay

train_loader, val_loader, class2ind, n_classes = load_data(DATA_DIR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, CLASS2IND_FILE, USE_OBJECTS,
                                                           USE_PARTS, False, None, NUM_WORKERS, MAX_SAMPLES)
# train_loader, val_loader, class2ind, n_classes = load_data(DATA_DIR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, NUM_WORKERS, MAX_SAMPLES)

model = get_FasterRCNN_model(n_classes).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

start_epoch = 0
best_val_mAP = 0.
if os.path.exists(model_save_path):
    print('Restoring trained model from %s' % model_save_path)
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']+1
    best_val_mAP = checkpoint['mAP']

for epoch in range(start_epoch, start_epoch + N_EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=500)
    _, stats = evaluate(model, val_loader, device=device, print_freq=1000, header='Val:')
    lr_scheduler.step()

    val_mAP = stats['bbox'][1] # AP @ IoU=0.5
    if val_mAP > best_val_mAP:
        best_val_mAP = val_mAP
        checkpoint = {'model': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
                      'epoch': epoch, 'mAP': val_mAP}
        torch.save(checkpoint, model_save_path)
    
    print('-'*100)

print('Best val_mAP: %.4f' % best_val_mAP)
