"""
Evaluate and Test
"""

import numpy as np
import torch
from PIR_Model import PIR_Model
from PIR_Dataset import PIR_Dataset

train_file = "../dataset/train.csv"
output_file = "../outputs/out.csv"
dataset = PIR_Dataset(train_file)
user_file = ["../model/user0.196863.pt", "../model/user0.199941.pt", "../model/user0.008831.pt"]
item_file = ["../model/item0.196863.pt", "../model/item0.199941.pt", "../model/item0.008831.pt"]
matrix = np.zeros((dataset.num_users, dataset.num_items))
for f1, f2 in zip(user_file, item_file):
    matrix += torch.matmul(torch.load(f1), torch.load(f2)).cpu().detach().numpy()
sum_true_pos = 0
sum_poss_pos = 0

with open(output_file, "w") as f:
    f.write("UserId,ItemId\n")
#     matrix = torch.matmul(model.user_matrix, model.item_matrix).cpu().detach().numpy()
    for i, row in enumerate(matrix):
        f.write("{},".format(i))
        rank = np.flip(np.argsort(row))
        pos_set = set(dataset.train_lists[i]+dataset.valid_lists[i])
        n_recom = 50
        sum_poss_pos += n_recom
        for idx in rank:
            if n_recom == 0:
                break
            if idx not in pos_set:
                n_recom -= 1
                f.write("{} ".format(idx))
            else:
                sum_true_pos += 1
                
        f.write("\n")            
print(sum_true_pos, sum_poss_pos)