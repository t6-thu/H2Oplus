import pdb
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2, device="cuda", scale=2, dropout=False):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.output_scale = scale
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        output = self.output_scale * torch.tanh(self.fc3(x))
        # output =self.fc3(x)
        return output

class ConcatDiscriminator(Discriminator):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

# class RatioEstimator(nn.Module):
#     def __init__(self, num_input, num_hidden, num_output=2, device="cuda", dropout=False):
#         super().__init__()
#         self.device = device
#         self.fc1 = nn.Linear(num_input, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, num_hidden)
#         self.fc3 = nn.Linear(num_hidden, num_output)

#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.tensor(x, dtype=torch.float).to(self.device)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         output = torch.sigmoid(self.fc3(x))
#         return output

class RatioEstimator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output=2, device="cuda", scale=1, dropout=False, output_activation=None):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_output)
        self.output_scale = scale
        self.output_activation = output_activation
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = F.relu(self.dropout_layer(self.fc1(x)))
            x = F.relu(self.dropout_layer(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        if self.output_activation:
            output = self.output_scale * self.output_activation(self.fc3(x))
        else:
            output = self.output_scale * self.fc3(x)
        
        return output
    
class ConcatRatioEstimator(RatioEstimator):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """
    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)