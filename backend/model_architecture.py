import torch
import torch.nn as nn

class NewsClassifierNN(nn.Module):
    def __init__(self, input_dim):
        super(NewsClassifierNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.abs(self.linear1(x))      # Absolute value activation
        x = torch.relu(self.linear2(x))     # ReLU activation
        x = self.linear3(x)                 # Output layer
        return x
