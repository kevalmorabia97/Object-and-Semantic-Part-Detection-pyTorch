import numpy as np
import os
import torch

from datasets import load_data, OBJECT_CLASSES
from models import get_FasterRCNN_model
from references.detection.engine import train_one_epoch, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA_DIR = 'data/VOCdevkit/VOC2010/'
N_CLASSES = len(OBJECT_CLASSES) # background class should also be counted!
BATCH_SIZE = 2

train_loader, val_loader = load_data(DATA_DIR, BATCH_SIZE)

model = get_FasterRCNN_model(N_CLASSES).to(device)

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.005)
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.005)

N_EPOCHS = 20
for epoch in range(1, N_EPOCHS+1):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
    evaluate(model, val_loader, device=device, print_freq=10000)