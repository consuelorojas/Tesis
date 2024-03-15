import torch
import torch.nn as nn

class AirModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1,hidden_size =  100, batch_first = True)
        self.linear = nn.Linear(100,1)

    def forward(self, x):
        x, _ = self.lstm(x) # x are the hidden states, _ is the lstm memory cell
        x = self.linear(x)
        return x
    
class AirModelLayers(nn.Module):
    def __init__(self, n_layers, dropout = None):
        super().__init__()
        self.dropout = dropout
        self.num_layers = n_layers
        self.lstm = nn.LSTM(input_size = 1,hidden_size =  100, num_layers = self.num_layers, batch_first = True, dropout = self.dropout)
        self.linear = nn.Linear(100,1)

    def forward(self, x):
        x, _ = self.lstm(x) # x are the hidden states, _ is the lstm memory cell
        x = self.linear(x)
        return x
    

class AirModel_DropOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = 1,hidden_size =  100, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(100,1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x, _ = self.lstm(x) # x are the hidden states, _ is the lstm memory cell
        x = self.dropout(x)
        x = self.linear(x)
        return x
    

class StackLSTM(nn.Module):
    def __init__(self, n_features, seq_len, output_size):
        super(StackLSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.output_size = output_size
        #pre-defined
        self.hidden_size = 500
        self.num_layers = 1

        self.lstm1 = nn.LSTM(input_size = self.n_features,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True)
        
        self.lstm2 = nn.LSTM(input_size = self.hidden_size,
                             hidden_size = self.output_size,
                             num_layers = self.num_layers,
                             batch_first = True)
        
        self.linear = nn.Linear(self.output_size*self.seq_len, self.output_size)

    def forward(self, x):
        batch_size, _, _ = x.size()

        #first layer
        hidden_state =  torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h1 = (hidden_state, cell_state)

        lstm_out, self.hidden1 = self.lstm1(x, h1)

        #second layer
        hidden_state =  torch.zeros(self.num_layers, batch_size, self.output_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.output_size)
        h2 = (hidden_state, cell_state)

        lstm_out, _ = self.lstm2(lstm_out, h2)

        #output layer
        x = lstm_out.contiguous().view(batch_size, -1) #flatten the output
        x = self.linear(x)
        return x
    

class LSTM(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(LSTM, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layers = num_layers

        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):

        #hidden state
        h0 = torch.zeros(self.layers, self.hid_dim)
        c0 = torch.zeros(self.layers, self.hid_dim)

        #forward prop
        out, (h, c) = self.lstm(x, (h0, c0))

        out = self.fc(out)

        return out
    