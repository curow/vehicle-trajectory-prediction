import torch
import torch.nn as nn

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
