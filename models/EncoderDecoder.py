import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
                 batch_size = 128,
                 input_size = 2,
                 embedding_size = 8,
                 hidden_size = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.linear = nn.Linear(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = F.relu(self.linear(input))
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        # hidden size is (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(1, self.batch_size, self.hidden_size)

class Decoder(nn.Module):
    def __init__(self,
                 batch_size = 128,
                 embedding_size = 8,
                 hidden_size = 16,
                 output_size = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = F.relu(self.linear1(input))
        output, hidden = self.gru(embedded, hidden)
        output = self.linear2(output)
        return output, hidden

    def init_hidden(self):
        # hidden size is (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(1, self.batch_size, self.hidden_size)
