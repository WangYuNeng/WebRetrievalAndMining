"""
PIR_Loss
"""
import torch
import numpy as np
import random

class BPR_Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.m = torch.nn.LogSigmoid()
    
    def forward(self, x, y):
        pos = x[:len(x)//2]
        neg = x[len(x)//2:]
        loss = -torch.mean(self.m((torch.clamp(pos-neg, min=-20, max=20))))
        return loss

"""
Calculate F1/MAP score
"""

def calc_score(dataset, model):
    with torch.no_grad():
        matrix = torch.matmul(model.user_matrix, model.item_matrix).cpu().detach().numpy()
        sum_AP = 0
        for i, row in enumerate(matrix):
            if dataset.is_train:
                golden_set = set(dataset.train_lists[i])
            else:
                golden_set = set(dataset.valid_lists[i])
                row[dataset.train_lists[i]] = np.min(row)
            rank = np.flip(np.argsort(row))
            n_recom = 50
            AP = 0
            true_pos = 0
            n_pos = len(golden_set)
            for i, idx in enumerate(rank[:n_recom]):
                if idx in golden_set:
                    true_pos += 1
                    AP += (true_pos/(i+1))/n_pos
            sum_AP += AP
    return sum_AP/len(dataset)
