from torch import nn

import utils
from sim_info import SimInfo

from utils import *

# constants
info = SimInfo()


# model
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

# consider our outputs
# gas, brake (allow both), steering angle

# consider our inputs:
# current position, velocity, slip on each wheel, steering angle(?)

# train the ai so that it maximizes the velocity relative to ideal spline.
# this means maximize the delta of "normalized car position" every moment

# fail the ai if it doesn't move for 3 seconds.


# start by waiting for 5 seconds to see starting position. 

# utils.parse_input is 1d
init = utils.parse_input()

dim = len(init)

model = LSTM(input_dim=dim, num_layers=2)
print(model)

while True:
    model.learn()
    inputs = utils.parse_input()

