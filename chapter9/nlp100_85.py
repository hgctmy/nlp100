from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import nlp100_80
import json

vectors = KeyedVectors.load_word2vec_format('../chapter8/GoogleNews-vectors-negative300.bin.gz', binary=True)

with open("word_id_dict.json")as f:
    word_id_dict = json.load(f)

# 学習済みベクトル
weights = torch.tensor(np.array([vectors[word] if word in vectors else np.random.normal(size=300) for word in word_id_dict.keys()]), dtype=torch.float)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_table(file_path)
        # 変換
        self.feature = data['TITLE']
        self.labels = torch.tensor([0 if x == 'b' else 1 if x == 't' else 2 if x == 'e' else 3 for x in data['CATEGORY']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(nlp100_80.id_sequence(self.feature[idx], word_id_dict))
        return feature, self.labels[idx]


class LSTM(nn.Module):
    def __init__(self, embedding_length, output_size, hidden_size, emb_weights, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(emb_weights)  # (単語id数+1,埋め込み次元数)
        self.bilstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, num_layers=2, bidirectional=True, dropout=dropout)  # 双方向，2層
        self.label = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, x_len):
        embedding = self.embedding(x)
        embedding = rnn.pack_padded_sequence(embedding, x_len.cpu(), batch_first=True, enforce_sorted=False)
        self.bilstm.flatten_parameters()
        _, hidden = self.bilstm(embedding, None)
        out = torch.cat([hidden[0][0], hidden[0][1]], dim=1)
        out = self.label(out)
        return out


model = LSTM(300, 4, 50, weights, 0)
learning_rate = 1e-2
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def collate_fn(batch):
    X, y = list(zip(*batch))
    x_len = torch.tensor([len(x) for x in X])
    X = list(rnn.pad_sequence(X, batch_first=True))
    X = torch.stack(X)
    y = torch.stack(y)
    return X, y, x_len


train_data = CustomDataset("../chapter8/train.txt")
test_data = CustomDataset("../chapter8/test.txt")
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

# GPUにする
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = torch.nn.DataParallel(model, device_ids=[0, 1])  # マルチGPUになるように
model.to(device)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y, x_len) in enumerate(dataloader):
        X, y, x_len = X.to(device), y.to(device), x_len.to(device)
        # 予測と損失の計算
        pred = model(X, x_len)
        loss = loss_fn(pred, y)
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータの最適化
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def accuracyandloss(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    loss, correct = 0, 0
    with torch.no_grad():  # 勾配計算をしない
        for X, y, x_len in dataloader:
            X, y, x_len = X.to(device), y.to(device), x_len.to(device)
            pred = model(X, x_len)
            loss += loss_fn(pred, y).item()  # 損失
            correct += (torch.argmax(pred, dim=1) == y).sum().item()  # predとyの一致する要素の数
        accuracy = correct / size
        loss = loss / size
    print(f'test_loss: {loss}', f'test_accuracy: {accuracy}')


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accuracyandloss(test_dataloader, model, loss_fn)
print("Done!")

'''
test_loss: 0.010266701119771782 test_accuracy: 0.7608695652173914
'''
