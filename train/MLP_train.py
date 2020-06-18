import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random
import time
import os

from data import get_dataset # custom helper function to get dataset

BATCH_SIZE = 16
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=6)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """Expected input is (batch_size, 20, 2)
    20: input sequence length
    2: the dimension of input feature (x and y)
    output shape: (batch_size, 30 * 2)
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 30 * 2)
        )
    
    def forward(self, x):
        # convert (batch_size, 20, 2) to (batch_size, 20 * 2)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model.to(dev)

epoches = 20 
print("start training...")
for epoch in range(epoches):
    start = time.time()
    model.train()
    for i, (xb, yb) in enumerate(train_loader):
        xb = xb.to(dev)
        yb = yb.to(dev).view(yb.size(0), -1)
        
        optimizer.zero_grad()
        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print("epoch {}, round {}/{} train loss: {:.4f}".format(epoch, i, len(train_loader), loss.item()))
            
    model.eval()
    model_dir = "saved_model/MLP"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/MLP_epoch{}".format(epoch))
    print("start validating...")
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb.to(dev)), yb.to(dev).view(yb.size(0), -1)) for xb, yb in val_loader)
    print("epoch {}, val loss: {:.4f}, time spend: {}s".format(
            epoch, val_loss / len(val_loader), time.time() - start))
