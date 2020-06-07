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
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-iter', type=int, default=1000)
parser.add_argument('-seed', type=int, default=428)
parser.add_argument('-validation_split', type=float, default=0.1)
parser.add_argument('-num_topic', type=int, default=64)
parser.add_argument('-use_bpr', type=int, default=1)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-l2_penalty', type=float, default=1e-6)
parser.add_argument('-use_wandb', type=int, default=0)
args = parser.parse_args()

if args.use_wandb:
    with open(".API_KEY", "r") as f:
        KEY = f.read()
    wandb.login(key=KEY)
    wandb.init(project="wm2020_pa2", config=vars(args))

if args.num_topic >= 64:
    torch.cuda.set_device(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_file = "../dataset/train.csv"
ITER = args.iter
torch.manual_seed(args.seed)
random.seed(args.seed)
dataset = PIR_Dataset(train_file, args.validation_split)
model = PIR_Model(dataset.num_users, dataset.num_items, args.num_topic)
if args.use_bpr == 1:
    loss_func = BPR_Loss()
else:
    loss_func = BCEWithLogitsLoss()
optimizer = optim.Adam([model.user_matrix, model.item_matrix], lr=args.lr, weight_decay=args.l2_penalty)

prev_valid_loss = 10000
valid_tolerance = 3
idxs = [i for i in range(len(dataset))]


for i in range(ITER):
    if valid_tolerance == 0:
        break
    train_loss = 0
    valid_loss = 0
    random.shuffle(idxs)
    dataset.set_is_train(True)
    print("Iter {}...".format(i))
    for uid in tqdm(idxs):
        optimizer.zero_grad()
        s1, s2, target = dataset[uid]
        output = model(uid, s1, s2)
        loss = loss_func(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_MAP = calc_score(dataset, model)
    dataset.set_is_train(False)
    for uid in idxs:
        with torch.no_grad():
            s1, s2, target = dataset[uid]
            output = model(uid, s1, s2)
            loss = loss_func(output, target)
            valid_loss += loss.item()
    valid_MAP = calc_score(dataset, model)
    print("iter {}: training loss={:6.4f}, MAP={:6.4f}; validation loss={:6.4f}, MAP={:6.4f}\n" \
          .format(i, train_loss/len(idxs), train_MAP, valid_loss/len(idxs), valid_MAP))

    if args.use_wandb:
        wandb.log({"Train Loss": train_loss/len(idxs), "Validation Loss": valid_loss/len(idxs),
                "Train MAP": train_MAP, "Validation MAP": valid_MAP})
    
    if valid_loss < prev_valid_loss:
        if args.use_wandb:
            torch.save(model.user_matrix, os.path.join(wandb.run.dir, "user_{:5.3f}_{:5.3f}_{}.pt".format(valid_MAP, valid_loss/len(idxs), i)))
            torch.save(model.item_matrix, os.path.join(wandb.run.dir, "item_{:5.3f}_{:5.3f}_{}.pt".format(valid_MAP, valid_loss/len(idxs), i)))
        prev_valid_loss = valid_loss
        valid_tolerance = 4
    else:
        valid_tolerance -= 1
