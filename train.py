import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models import *


def remove_punct(verse):
    verse = verse.translate(str.maketrans('', '', punctuation))
    verse = verse.lower()
    return verse


class LyricDataset(Dataset):
    def __init__(self, df, cond=None, seq_len=5):
        self.data = df.lyric.to_list()
        self.category = list(df.artist.unique())
        self.encode_cat = {key: val for val, key in enumerate(self.category)}
        self.stoi = {**{'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3},
                     **{j: i for i, j in enumerate(set([word for word in ' '.join(self.data).split()]), start=4)}}
        self.itos = {**{0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'},
                     **{i: j for i, j in enumerate(set([word for word in ' '.join(self.data).split()]), start=4)}}
        self.full_text = ' '.join(self.data)
        self.full_idx = [self.stoi[word] for word in self.full_text.split()]
        self.seq_len = seq_len
        self.cond = cond
        self.full_artists = self.artist_list(df)

    # Bars are split by seq_len; attach artist to each possible sliding window.
    def artist_list(self, df):
        artist_list = []
        for artist in self.category:
            subdf = df[df.artist == artist]
            text = ' '.join(subdf.lyric.to_list()).split()
            artist_list += [artist] * len(text)
        return artist_list

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, idx):
        cat = self.encode_cat[self.full_artists[idx]]
        input = torch.tensor(self.full_idx[idx:idx + self.seq_len])
        target = torch.tensor(self.full_idx[idx + 1:idx + self.seq_len + 1])
        return cat, input, target


def RNN_train(model, dataset, batch_dim=32, lr=1e-3, epochs=15, cond=None):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_dim)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    start = time()

    losses = []
    for epoch in range(epochs):
        h, c = model.init_hidden()

        for batch, (cat, input, target) in enumerate(loader):
            val = cat.data[0]
            cat_tens = torch.full_like(input, val)
            # print(batch, input.shape, cat_tens.shape)
            cat_tens = cat_tens.type(torch.FloatTensor)
            cat_tens = torch.unsqueeze(cat_tens, 2).to('cuda')
            input = input.to('cuda')
            target = target.to('cuda')

            opt.zero_grad()
            pred, (h, c) = model(cat_tens, input, (h, c), cond=cond)
            loss = crit(pred.transpose(1, 2), target)

            loss.backward()
            opt.step()

        elapsed = time() - start
        losses.append(loss.item())
        print(f'epoch: {epoch + 1}, loss: {loss.item()}, elapsed: {elapsed:.2f} s')
    return losses


def FiLMed_train(model, dataset, batch_dim=32, lr=1e-3, epochs=15):
    model.train()
    loader = DataLoader(dataset, batch_size=batch_dim)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    start = time()
    losses = []
    for epoch in range(epochs):
        h, c = model.init_hidden()

        for batch, (cat, input, target) in enumerate(loader):
            val = cat.data[0]
            cat_tens = torch.full_like(input, val)
            cat_tens = cat_tens.type(torch.FloatTensor)
            cat_tens = torch.unsqueeze(cat_tens, 2).to('cuda')
            input = input.to('cuda')
            target = target.to('cuda')

            opt.zero_grad()
            pred, (h, c) = model(cat_tens, input, (h, c))
            loss = crit(pred.transpose(1, 2), target)

            loss.backward()
            opt.step()

        losses.append(loss.item())
        elapsed = time() - start
        print(f'epoch: {epoch + 1}, loss: {loss.item()}, elapsed: {elapsed:.2f} s')
    return losses


def GAN_train(gen, dis, dataset, batch_dim=32, lr=0.000002, epochs=15, cond=None, film=False):
    gen.train()
    dis.train()
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    dis_opt = optim.Adam(dis.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_dim)
    crit = nn.BCELoss()

    high = len(dataset) - 1

    # Given random noise, generate (batch_dim, 5) size output tensor.
    def fake_gen(cat_tens, noise, h, c):
        tokens = noise
        for j in range(5):
            sub = tokens[:, j:j + 5].long()
            out_prob, (h, c) = gen(cat_tens, sub, (h, c), cond=cond)
            token_prob = F.softmax(out_prob[:, -1, :], dim=0).to('cuda')
            new = torch.multinomial(token_prob, 1).to('cuda')
            tokens = torch.cat((tokens, new), 1).to('cuda')
        fake = tokens[:, -5:].to('cuda')
        return fake

    start = time()

    gen_losses = []
    dis_losses = []
    for epoch in range(epochs):
        h, c = gen.init_hidden()

        for batch, (cat, input, target) in enumerate(loader):
            val = cat.data[0]
            cat_tens = torch.full((batch_dim, 5, 1), val).type(torch.FloatTensor).to('cuda')
            input = input.to('cuda')
            target = target.to('cuda')

            noise = torch.randint(low=0, high=high, size=(batch_dim, 5)).to('cuda')
            fake = fake_gen(cat_tens, noise, h, c)

            # Train disc
            dis_opt.zero_grad()
            pred_r, (h, c) = dis(input, (h, c))
            loss_r = crit(pred_r, torch.ones_like(pred_r).to('cuda'))
            loss_r.backward()

            pred_f, (h, c) = dis(fake, (h, c))
            loss_f = crit(pred_f, torch.zeros_like(pred_f).to('cuda'))
            loss_f.backward()

            dis_opt.step()
            d_loss = loss_r + loss_f

            # Train gen
            noise = torch.randint(low=0, high=high, size=(batch_dim, 5)).to('cuda')
            fake = fake_gen(cat_tens, noise, h, c)

            gen_opt.zero_grad()
            pred, (h, c) = dis(fake, (h, c))
            g_loss = crit(pred, torch.ones_like(pred).to('cuda'))
            g_loss.backward()
            gen_opt.step()

        elapsed = time() - start
        gen_losses.append(g_loss.item())
        dis_losses.append(d_loss.item())
        print(f'epoch: {epoch + 1}, gen_loss: {g_loss.item()},  dis_loss: {d_loss.item()}, elapsed: {elapsed:.2f} s')
    return gen_losses, dis_losses


def create_lyric(model, dataset, input, artist, bar_len=15):
    model.eval()
    words = input.split()
    artist = dataset.encode_cat[artist]
    h, c = model.init_hidden()
    tokens = np.array([dataset.stoi[word] for word in input.split()])

    def predict(model, dataset, tensor, artist, h, c):
        cat_tens = torch.full_like(tensor, artist)
        cat_tens = cat_tens.type(torch.FloatTensor)
        cat_tens = torch.unsqueeze(cat_tens, 2).to('cuda')
        out, (h, c) = model(cat_tens, tensor, (h, c))
        token_prob = F.softmax(out[0][2], dim=0).detach().cpu().numpy()
        rand_idx = np.random.choice(len(dataset), p=token_prob)
        return dataset.itos[rand_idx], (h, c)

    for i in range(bar_len):
        sub = words[i:i + dataset.seq_len]
        pred_word, (h, c) = predict(model, dataset, torch.tensor([[dataset.stoi[word] for word in sub]]).to('cuda'),
                                    artist, h, c)
        words.append(pred_word)

    return ' '.join(words)


def gen_batch(dataset, model, cat_tens, h, c, dim, cond=None):
    noise = torch.randint(low=0, high=len(dataset), size=(dim, 5)).to('cuda')
    for j in range(5):
        sub = noise[:, j:j + 5].long()
        out_prob, (h, c) = model(cat_tens, sub, (h, c), cond=cond)
        token_prob = F.softmax(out_prob[:, -1, :], dim=0).to('cuda')
        new = torch.multinomial(token_prob, 1).to('cuda')
        noise = torch.cat((noise, new), 1).to('cuda')
    generated = noise[:, -5:].to('cuda')
    return generated


def fid(tens1, tens2):
    m1 = torch.mean(tens1)
    m2 = torch.mean(tens2)
    dist = torch.pow((m1 - m2), 2)
    cov1 = torch.cov(tens1)
    cov2 = torch.cov(tens2)
    tr = torch.trace(cov1 + cov2 - 2 * torch.sqrt(cov1 * cov2))
    return dist.item() + tr.item()


def main():
    train_df = pd.read_csv('train.csv')
    train_df.lyric = train_df.lyric.apply(lambda x: remove_punct(x))

    test_df = pd.read_csv('test.csv')
    test_df.lyric = test_df.lyric.apply(lambda x: remove_punct(x))

    dataset = LyricDataset(train_df)

    # Train RNN models
    LSTM_model_add = LSTM_Model(dataset)
    LSTM_model_add.to('cuda')
    LSTM_add_loss = RNN_train(LSTM_model_add, dataset, cond='add', epochs=15)

    LSTM_model_mul = LSTM_Model(dataset)
    LSTM_model_mul.to('cuda')
    LSTM_mul_loss = RNN_train(LSTM_model_mul, dataset, cond='mul', epochs=15)

    GRU_model_add = GRU_Model(dataset)
    GRU_model_add.to('cuda')
    GRU_add_loss = RNN_train(GRU_model_add, dataset, cond='add', epochs=15)

    GRU_model_mul = GRU_Model(dataset)
    GRU_model_mul.to('cuda')
    GRU_mul_loss = RNN_train(GRU_model_mul, dataset, cond='mul', epochs=15)

    FiLMed_GRU = FiLMed_GRU_Model(dataset)
    FiLMed_GRU.to('cuda')
    GRU_FiLMed_losses = FiLMed_train(FiLMed_GRU, dataset, epochs=15)

    FiLMed_LSTM = FiLMed_LSTM_Model(dataset)
    FiLMed_LSTM.to('cuda')
    LSTM_FiLMed_losses = FiLMed_train(FiLMed_LSTM, dataset, epochs=15)

    # Train GAN models
    batch_dim = 16
    gen_add = Generator(dataset, batch_dim=batch_dim)
    dis_add = Discriminator(dataset, batch_dim=batch_dim)
    gen_add.to('cuda')
    dis_add.to('cuda')
    GEN_add_loss, DIS_add_loss = GAN_train(gen_add, dis_add, dataset, cond='add', epochs=15, batch_dim=batch_dim)

    gen_mul = Generator(dataset, batch_dim=batch_dim)
    dis_mul = Discriminator(dataset, batch_dim=batch_dim)
    gen_mul.to('cuda')
    dis_mul.to('cuda')
    GEN_mul_loss, DIS_mul_loss = GAN_train(gen_mul, dis_mul, dataset, cond='mul', epochs=15, batch_dim=batch_dim)

    gen_FiLMed = FiLMed_Generator(dataset, batch_dim=batch_dim)
    dis_FiLMed = Discriminator(dataset, batch_dim=batch_dim)
    gen_FiLMed.to('cuda')
    dis_FiLMed.to('cuda')
    GENFILM_loss, DISFILM_loss = GAN_train(gen_FiLMed, dis_FiLMed, dataset, cond=None, epochs=15, batch_dim=batch_dim)

    test_set = LyricDataset(test_df)
    test_list = [tup[1] for tup in list(test_set)]
    test_tens = torch.reshape(torch.cat(test_list), (-1, 5)).double()[:10000, :].to('cuda')

    model = gen_FiLMed  # Should be swapped out
    val = random.choice([0, 1, 2])
    cat_tens = torch.full((10000, 5, 1), val).type(torch.FloatTensor).to('cuda')
    h, c = model.init_hidden()
    generated = gen_batch(dataset, model, cat_tens, h, c, dim=10000, cond=None)
    gen_tens = generated.double()

    fid = fid(test_tens, gen_tens)

    # del model
    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
