import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 8,
                 hidden_size = 16,
                 n_layers = 4,
                 dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout, batch_first = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [batch size, sequence len, feature size]
        for the argoverse trajectory data, size(x) is [batch size, 20, 2]
        """
        # embedded: [batch size, sequence len, embedding size]
        embedded = self.dropout(self.linear(x))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(embedded)
        # size(hidden): [n layers * n directions, batch size, hidden size]
        # size(cell): [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 8,
                 hidden_size = 16,
                 n_layers = 4,
                 dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, input, hidden):
        embedded = F.relu(self.linear1(input))
        output, hidden = self.gru(embedded, hidden)
        output = self.linear2(output)
        return output, hidden

    def init_hidden(self):
        # hidden size is (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(1, self.batch_size, self.hidden_size)

class EncoderDecoder(nn.Module):

