import torch.nn as nn
import torch
import os


class simple_CNN(nn.Module):
    def __init__(self, using_data_num, use_10k_embed=None, device='cpu'):
        super(simple_CNN, self).__init__()
        self.device = device
        input_dim = 100
        output_dim = 1

        if use_10k_embed is not None:
            self.embedding = nn.Embedding.from_pretrained(use_10k_embed)
        else:
            self.embedding = nn.Embedding(10002, input_dim)

        self.layers = nn.Sequential(
            nn.Conv1d(100, 100, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(100, 100, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(100, 100, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(100, 100, 3),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(100, output_dim)

    def forward(self, x_in):
        x_in = x_in.to(self.device)
        x_in = self.embedding(x_in)
        print(x_in.shape)
        x_in = x_in.transpose(1, 2) # (N, C, L)
        x = self.layers(x_in)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        return x


class simple_LSTM(nn.Module):
    def __init__(self, lstm_length, hidden_dim=100, pretrained_embedding=None, device='cpu'):
        super(simple_LSTM, self).__init__()

        self.device = device

        input_dim = 100
        output_dim = 1

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding = nn.Embedding(10002, input_dim)

        self.encoder = nn.LSTMCell(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.lstm_length = lstm_length
        self.tanh = nn.Tanh()

    def forward(self, x_in):
        batch_size = x_in.size()[0]
        hx = torch.randn(batch_size, self.hidden_dim)
        cx = torch.randn(batch_size, self.hidden_dim)

        x_in = x_in.to(self.device)
        hx, cx = hx.to(self.device), cx.to(self.device)

        x_in = self.embedding(x_in.T)
        output = []
        for i in range(x_in.size()[0]):
            hx, cx = self.encoder(x_in[i], (hx, cx))
            hx = self.tanh(hx)
            output.append(hx)
        output = output[-1]
        outs = self.decoder(output)
        return outs


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def load_embedding():
    weights = []

    with open(os.path.join('.', 'data', '10k_embed.txt')) as f:
        embedding = f.readlines()

        for line in embedding:
            rows = line.split()[1:]
            rows = list(map(float, rows))
            weights.append(rows)

    weights.append([-1.] * 100)
    weights.append([0.] * 100)

    weights = torch.FloatTensor(weights)

    return weights
