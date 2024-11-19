import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers = 1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, x):
        lstm_out, status = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :]) 
        return output
