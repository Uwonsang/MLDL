import torch.nn as nn
import torch


class CNNBasedModel(nn.Module):
    def __init__(self, max_length, pretrained_embedding=None, device='cpu'):
        super(CNNBasedModel, self).__init__()

        self.device = device

        input_dim = 100
        output_dim = 1

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
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
        x_in = x_in.transpose(1, 2) # (N, C, L)
        x = self.layers(x_in)

        batch_size = x.shape[0]

        x = x.reshape(batch_size, -1)

        x = self.fc1(x)
        return x


class LSTMBasedModel(nn.Module):
    def __init__(self, lstm_length, hidden_dim=100, pretrained_embedding=None, device='cpu'):
        super(LSTMBasedModel, self).__init__()

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


class BiLSTMModel(nn.Module):
    def __init__(self, lstm_length, hidden_dim=100, pretrained_embedding=None, device='cpu'):
        super(BiLSTMModel, self).__init__()

        self.device = device

        input_dim = 100
        output_dim = 1

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        else:
            self.embedding = nn.Embedding(10002, input_dim)

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=4, bidirectional=True)
        self.decoder = nn.Linear(4 * hidden_dim, output_dim)

    def forward(self, x_in):
        x_in = x_in.to(self.device)
        x_in = self.embedding(x_in.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(x_in)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
