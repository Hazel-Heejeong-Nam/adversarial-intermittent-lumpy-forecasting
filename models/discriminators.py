import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LSTM_Discriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, batch_size, device, num_feats, hidden_units=256, drop_prob=0.6):

        super(LSTM_Discriminator, self).__init__()

        # params
        self.hidden_dim = hidden_units
        self.num_layers = 2
        self.device = device

        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2*hidden_units), out_features=1)
        
        # self.state = self.init_hidden(batch_size)

    def forward(self, x, state):
        ''' Forward prop
        '''
        # x: (batch_size, seq_len, num_feats)
        drop_in = self.dropout(x) # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(drop_in.reshape(x.shape[0], -1, 1), state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)

        return out , lstm_out

    def init_hidden(self, batch_size, device):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2 # for being bidirectional
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                        self.hidden_dim).zero_().to(device),
            weight.new(self.num_layers * layer_mult, batch_size,
                        self.hidden_dim).zero_().to(device))
        
        return hidden

    
class MLP_Discriminator(nn.Module):
    def __init__(self, shape, hidden_units=16):
        super(MLP_Discriminator, self).__init__()
        self.hidden_units = hidden_units
        self.model = nn.Sequential(
            nn.Linear(shape, self.hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        data = data.view(data.size(0), -1)
        validity = self.model(data)
        validity = torch.clamp(validity, min=0.001, max=0.999)
        return validity
    
