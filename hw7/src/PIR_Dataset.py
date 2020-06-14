"""
PIR_Dataset
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from random import sample, shuffle

C1 = 1

class PIR_Dataset(Dataset):

    def __init__(self, csv_file, valid_split=0.1):
        user_dataframe = pd.read_csv(csv_file)
        self.train_lists, self.valid_lists = self._get_pos_lists(user_dataframe, valid_split)
        self.num_items = max([max(l) for l in self.train_lists])+1
        self.num_users = len(self.train_lists)
        self.neg_lists = self._get_neg_lists()
        self.train_targets, self.valid_targets = self._get_targets()
        self.is_train = True
        
    def set_is_train(self, is_train):
        self.is_train = is_train
    
    def __len__(self):
        return len(self.train_lists)

    def __getitem__(self, idx):
        if self.is_train == True:
            s1 = self.train_lists[idx]
            s2 = sample(self.neg_lists[idx], C1*len(self.train_lists[idx]))
            return s1, s2, self.train_targets[idx]
        else:
            # might sample neg_data from validation set but ignore for now
            s1 = self.valid_lists[idx]
            s2 = sample(self.neg_lists[idx], C1*len(self.valid_lists[idx]))
            return s1, s2, self.valid_targets[idx]
    
    def _get_pos_lists(self, df, valid_split):
        tot_lists = [list(map(int, item_str.split())) for item_str in df["ItemId"]]
        train_lists, valid_lists = [], []
        for l in tot_lists:
            shuffle(l)
            idx = int(len(l)*valid_split) + 1
            valid_lists.append(sorted(l[-idx:]))
            train_lists.append(sorted(l[:-idx]))
        return train_lists, valid_lists
    
    def _get_neg_lists(self):
        neg_lists = [[] for _ in range(self.num_users)]
        for user_id in range(self.num_users):
            pos_iter = 0
            for item_id in range(self.num_items):
                if item_id == self.train_lists[user_id][pos_iter]:
                    pos_iter += (pos_iter != len(self.train_lists[user_id])-1)
                else:
                    neg_lists[user_id].append(item_id)
        return neg_lists
    
    def _get_targets(self):
        return [torch.cat((torch.ones(len(l)), torch.zeros(C1*len(l)))) for l in self.train_lists], \
                [torch.cat((torch.ones(len(l)), torch.zeros(C1*len(l)))) for l in self.valid_lists]
