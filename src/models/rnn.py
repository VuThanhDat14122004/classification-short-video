import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, CNNPretrained, hidden_size, num_classes, batch_size):
        super(RNN, self).__init__()
        self.CNNPretrained = CNNPretrained
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.rnn = nn.RNN(
            input_size=(2048 * 4 * 4),
            hidden_size=hidden_size,
            num_layers=4,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        input = list()
        for index in range(x.size(1)):
            frame = x[:, index]

            frame = frame.float()
            with torch.no_grad():
                _x = self.CNNPretrained(frame)
                _x = _x.view(_x.size(0), 2048 * 4 * 4)
            input.append(_x)
        input = torch.stack(input, dim=1)

        out1, hidden1 = self.rnn(input)
        res = self.fc1(hidden1[0])
        x = self.fc2(res)

        return x
