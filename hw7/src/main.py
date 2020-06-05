"""
PIR_Trainer
"""

import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import random
import os
from PIR_Model import PIR_Model
from PIR_Dataset import PIR_Dataset
from PIR_utils import BPR_Loss, calc_score


train_file = "../dataset/train.csv"
ITER = 1000
torch.manual_seed(66)
random.seed(66)
checkpoint_dir = "../model/"
dataset = PIR_Dataset(train_file, 0.1)
model = PIR_Model(dataset.num_users, dataset.num_items, 64)
loss_func = BPR_Loss()
# loss_func = BCEWithLogitsLoss()
optimizer = optim.Adam([model.user_matrix, model.item_matrix], lr=1e-3, weight_decay=1e-6)

prev_valid_loss = 100000
valid_tolerance = 3
idxs = [i for i in range(len(dataset))]


for i in range(ITER):
    if valid_tolerance == 0:
        break
    train_loss = 0
    valid_loss = 0
    random.shuffle(idxs)
    dataset.set_is_train(True)
    print("training...")
    for uid in tqdm(idxs):
        optimizer.zero_grad()
        s1, s2, target = dataset[uid]
        output = model(uid, s1, s2)
        loss = loss_func(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_MAP = calc_score(dataset, model)
    print("validating...")
    dataset.set_is_train(False)
    for uid in tqdm(idxs):
        with torch.no_grad():
            s1, s2, target = dataset[uid]
            output = model(uid, s1, s2)
            loss = loss_func(output, target)
            valid_loss += loss.item()
    valid_MAP = calc_score(dataset, model)
    print("iter {}: training loss={:6.4f}, MAP={:6.4f}; validation loss={:6.4f}, MAP={:6.4f}\n" \
          .format(i, train_loss/len(idxs), train_MAP, valid_loss/len(idxs), valid_MAP))
    if valid_loss < prev_valid_loss:
        if prev_valid_loss != 100000:
            os.remove("{}user{:8.6f}.pt".format(checkpoint_dir, prev_valid_loss/len(idxs)))
            os.remove("{}item{:8.6f}.pt".format(checkpoint_dir, prev_valid_loss/len(idxs)))
        torch.save(model.user_matrix, "{}user{:8.6f}.pt".format(checkpoint_dir, valid_loss/len(idxs)))
        torch.save(model.item_matrix, "{}item{:8.6f}.pt".format(checkpoint_dir, valid_loss/len(idxs)))
        prev_valid_loss = valid_loss
        valid_tolerance = 3
    else:
        valid_tolerance -= 1
