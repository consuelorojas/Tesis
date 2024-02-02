import torch
import torch.nn as nn

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size = 1,hidden_size =  100, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(100,1)

    def forward(self, x):
        x, _ = self.lstm(x) # x are the hidden states, _ is the lstm memory cell
        x = self.linear(x)
        return x
    

class MultiLSTM(nn.Module):
    def __init__(self, n_features, seq_len, output_size):
        super(MultiLSTM, self).__init__()
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
        batch_size, seq_len, _ = x.size()

        hidden_state =  torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h1 = (hidden_state, cell_state)

        lstm_out, self.hidden1 = self.lstm1(x, h1) #first layer

        hidden_state =  torch.zeros(self.num_layers, batch_size, self.output_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.output_size)
        h2 = (hidden_state, cell_state)

        lstm_out, self.hidden2 = self.lstm2(lstm_out, h2) #second layer

        x = lstm_out.contiguous().view(batch_size, -1) #flatten the output

        #x, _ = self.lstm(x) # x are the hidden states, _ is the lstm memory cell
        x = self.linear(x)
        return x
    
