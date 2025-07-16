from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim = 24, hidden_dim = 12, output_dim = 8, activation_func = nn.Sigmoid(), fc_dim = 1, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.activation = activation_func
        self.fc = nn.Sequential()
        self.fc.add_module('fc0', nn.Linear(input_dim, fc_dim))
        self.fc.add_module("activation0", self.activation)
        for i in range(num_layers-1):
            self.fc.add_module("fc"+ str(i+1), nn.Linear(fc_dim, fc_dim))
            self.fc.add_module("activation" + str(i+1), self.activation)


    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        output, hidden = self.lstm(x, hidden)  # output: (batch, seq_len, hidden_dim)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden

model = LSTM(num_layers=2)
print(model)