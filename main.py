import argparse
import numpy as np
import os
import torch

from datasets import load_data
from models import get_FasterRCNN_model
from references.detection.engine import train_one_epoch, evaluate
from utils import set_all_seeds


########## CMDLINE ARGS ##########
parser = argparse.ArgumentParser('Train Model')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-dir', '--data_dir', type=str, default='data/VOCdevkit/VOC2010/')
parser.add_argument('-tr', '--train_split', type=str, default='train')
parser.add_argument('-val', '--val_split', type=str, default='val')
parser.add_argument('-cf', '--classes_file', type=str, default='object_classes')
parser.add_argument('-e', '--n_epochs', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
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
CLASSES_FILE = args.classes_file if not args.classes_file == '' else None
N_EPOCHS = args.n_epochs
USE_OBJECTS = bool(args.use_objects)
USE_PARTS = bool(args.use_parts)
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples if args.max_samples > 0 else None

model_save_path = 'saved_model.pth'

########## Hyperparameters ##########
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
WEIGHT_DECAY = args.weight_decay

train_loader, val_loader, classes = load_data(DATA_DIR, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, CLASSES_FILE, USE_OBJECTS,
                                              USE_PARTS, NUM_WORKERS, MAX_SAMPLES)
n_classes = len(classes) # background class should also be counted!

model = get_FasterRCNN_model(n_classes).to(device)
if os.path.exists(model_save_path):
    print('Restoring trained model from %s' % model_save_path)
    model.load_state_dict(torch.load(model_save_path))

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

for epoch in range(N_EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=500)
    evaluate(model, val_loader, device=device, print_freq=10000, header='Val:')

    ## TODO: Check if improvement, then only overwrite saved model
    torch.save(model.state_dict(), model_save_path)
