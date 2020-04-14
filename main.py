import numpy as np
import os
import torch

from datasets import PascalPartVOCDetection, get_transform, OBJECT_CLASSES, PART_CLASSES
from models import get_FasterRCNN_model
from references.detection.engine import train_one_epoch, evaluate
from references.detection.utils import collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA_DIR = 'data/VOCdevkit/VOC2010/'
N_CLASSES = len(OBJECT_CLASSES) # background class should also be counted!
BATCH_SIZE = 2

train_dataset = PascalPartVOCDetection(DATA_DIR, 'train', transforms=get_transform(is_train=True))
val_dataset = PascalPartVOCDetection(DATA_DIR, 'val', transforms=get_transform(is_train=False))

# train_dataset = torch.utils.data.Subset(train_dataset, np.arange(100)) # subset for quick run
# val_dataset = torch.utils.data.Subset(val_dataset, np.arange(100)) # subset for quick run

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_FasterRCNN_model(N_CLASSES).to(device)

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.005)
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.005)

N_EPOCHS = 20
for epoch in range(1, N_EPOCHS+1):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
    evaluate(model, val_loader, device=device, print_freq=10000)