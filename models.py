import torch

from torch import nn
from time import time


class GRU_Model(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(GRU_Model, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.gru_layer = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )
        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )

    def forward(self, cat, x, hidden, cond=None):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        if cond == 'add':
            out = torch.add(out, cout)
        elif cond == 'mul':
            out = torch.mul(out, cout)
        out, hid = self.gru_layer(out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class LSTM_Model(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(LSTM_Model, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.lstm_layer = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )

        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )

    def forward(self, cat, x, hidden, cond=None):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        if cond == 'add':
            out = torch.add(out, cout)
        elif cond == 'mul':
            out = torch.mul(out, cout)
        out, hid = self.lstm_layer(out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class FiLM_Generator(nn.Module):
    def __init__(self):
        super(FiLM_Generator, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 128 * 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig1(out)
        out = self.fc2(out)
        gamma = out[:, :, :128]
        beta = out[:, :, 128:]
        return gamma, beta


class FiLM_Layer(nn.Module):
    def __init__(self):
        super(FiLM_Layer, self).__init__()
        self.gen = FiLM_Generator()

    def forward(self, cout, x):
        gamma, beta = self.gen(cout)
        out = gamma * x + beta
        return out


class FiLMed_GRU_Model(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(FiLMed_GRU_Model, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.gru_layer = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )
        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )
        self.FiLM = FiLM_Layer()

    def forward(self, cat, x, hidden):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        out, hid = self.gru_layer(out)
        out = self.FiLM(cout, out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class FiLMed_LSTM_Model(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(FiLMed_LSTM_Model, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.lstm_layer = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )
        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )
        self.FiLM = FiLM_Layer()

    def forward(self, cat, x, hidden):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        out, hid = self.lstm_layer(out)
        out = self.FiLM(cout, out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class Generator(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(Generator, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.gru_layer = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )
        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )

    def forward(self, cat, x, hidden, cond=None):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        if cond == 'add':
            out = torch.add(out, cout)
        elif cond == 'mul':
            out = torch.mul(out, cout)
        out, hid = self.gru_layer(out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class FiLMed_Generator(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(FiLMed_Generator, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.gru_layer = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            self.vocab_dim
        )
        self.cond_layer = nn.Linear(
            1,
            128,
            bias=False
        )
        self.FiLM = FiLM_Layer()

    def forward(self, cat, x, hidden, cond=None):
        out = self.emb_layer(x)
        cout = self.cond_layer(cat)
        out, hid = self.gru_layer(out)
        out = self.FiLM(cout, out)
        out = self.lin_layer(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)


class Discriminator(nn.Module):
    def __init__(self, dataset, batch_dim=32):
        super(Discriminator, self).__init__()
        self.vocab_dim = len(dataset.stoi)
        self.seq_len = dataset.seq_len
        self.batch_dim = batch_dim

        self.emb_layer = nn.Embedding(
            num_embeddings=self.vocab_dim,
            embedding_dim=128
        )
        self.gru_layer = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        self.lin_layer = nn.Linear(
            128,
            1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        out = self.emb_layer(x)
        out, hid = self.gru_layer(out)
        out = self.lin_layer(out)
        out = self.sig(out)
        return out, hid

    def init_hidden(self):
        return torch.zeros(2, self.batch_dim, 128), torch.zeros(2, self.batch_dim, 128)
