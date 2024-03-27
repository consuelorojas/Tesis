import torch
import torch.nn as nn

# Recurrent Neural Network

class SimpleRNN(nn.Module):
    """
    A simple recurrent neural network (RNN) module.

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        output_size (int): The number of output features
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a `stacked RNN`, with the second RNN taking in outputs of the first RNN and producing the final results.

    Attributes:
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers
        rnn (nn.RNN): The RNN layer
        fc (nn.Linear): The fully connected layer

    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        out, _ = self.rnn(x, h0)
        
        out = self.fc(out)
        return out




# Multi-layer perceptron
    
    
class MLP(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.

    Args:
        in_size (int): The number of input features.
        hid_size (int): The number of hidden units in each hidden layer.
        out_size (int): The number of output units.

    Attributes:
        layers (nn.Sequential): The sequential layers of the MLP.

    Methods:
        forward(x): Performs forward pass through the MLP.

    '''
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, out_size)
        )

    def forward(self, x):
        return self.layers(x)
    
# Long Short-Term Memory Network
    
class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) module.

    Args:
        in_dim (int): The number of expected features in the input x.
        hid_dim (int): The number of features in the hidden state h.
        out_dim (int): The number of output features.
        num_layers (int): Number of recurrent layers.

    Attributes:
        in_dim (int): The number of expected features in the input x.
        hid_dim (int): The number of features in the hidden state h.
        out_dim (int): The number of output features.
        layers (int): Number of recurrent layers.
        lstm (nn.LSTM): LSTM layer.
        fc (nn.Linear): Fully connected layer.

    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout = False):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layers = num_layers
        self.drop = dropout

        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # hidden state
        #h0 = torch.zeros(self.layers, x.size(0), self.hid_dim)
        #c0 = torch.zeros(self.layers, x.size(0), self.hid_dim)
        h0 = torch.zeros(self.layers, self.hid_dim)
        c0 = torch.zeros(self.layers, self.hid_dim)

        # forward prop
        out, (h, c) = self.lstm(x, (h0, c0))
        
        if self.dropout:
            out = self.dropout(out)
            out = self.fc(out)
        else:
            out = self.fc(out)

        return out