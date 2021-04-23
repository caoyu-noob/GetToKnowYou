import torch

import torch.nn as nn
import torch.nn.functional as F

class PredicateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredicateModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_state):
        output_state = self.linear(input_state)
        return F.sigmoid(output_state)