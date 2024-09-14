import torch
import numpy
import torch.nn as nn
from torch.autograd import Variable

class MLP_forecaster(nn.Module):
    def __init__(self, in_shape, out_shape, num_layer=3, hidden_units=128):
        super(MLP_forecaster, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.num_layer = num_layer
        self.hidden_units = hidden_units
        self.model = self.init_model()
    
    def init_model(self):
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.in_shape, self.hidden_units))
        layers.append(nn.BatchNorm1d(self.hidden_units))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for _ in range(self.num_layer - 1):
            layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            layers.append(nn.BatchNorm1d(self.hidden_units))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_units, self.out_shape))
        return nn.Sequential(*layers)

    def forward(self, data):
        data = data.view(data.size(0), -1)
        validity = self.model(data)
        validity = torch.clamp(validity, min=0.001, max=0.999)
        return validity
    

class LSTM_forecaster(nn.Module):

    def __init__(self, seq_length, batch_size, device, num_classes=1, input_size=1, hidden_size=256, num_layers=2):
        super(LSTM_forecaster, self).__init__()
        
        self.num_classes = num_classes # 1 (하나의 forecasting period당 하나의 value라서 그런듯)
        self.num_layers = num_layers #2
        self.input_size = input_size # 1
        self.hidden_size = hidden_size # 256
        self.seq_length = seq_length
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, seq_length)
        
        self.state = self.init_hidden(batch_size)

    
    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 1 # uni-directional
        hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_size).zero_().to(self.device),
                    weight.new(self.num_layers * layer_mult, batch_size,
                                self.hidden_size).zero_().to(self.device))
        
        return hidden
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1, 1)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, self.state)
        h_out = h_out[1:,:,:].view(-1, self.hidden_size) 
        out = self.fc(h_out)
        
        return out