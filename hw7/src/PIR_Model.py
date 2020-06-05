"""
PIR_Model
"""
import torch

NUM_TOPICS = 256

class PIR_Model(torch.nn.Module):
    
    def __init__(self, num_users, num_items, topics=NUM_TOPICS):
        super().__init__()
        self.user_matrix = torch.randn((num_users, topics), requires_grad=True)
        self.item_matrix = torch.randn((topics, num_items), requires_grad=True)
        self.item_bias = torch.randn((num_items,), requires_grad=True)
        
    def forward(self, user_id, pos_idx, neg_idx):
        pos_tensor = torch.matmul(self.user_matrix[user_id], self.item_matrix[:,pos_idx]) + self.item_bias[pos_idx]
        neg_tensor = torch.matmul(self.user_matrix[user_id], self.item_matrix[:,neg_idx]) + self.item_bias[neg_idx]
        if pos_tensor.dim() == 1:
            return torch.cat((pos_tensor, neg_tensor))
        else:
            return torch.cat((pos_tensor, neg_tensor), axis=1)
