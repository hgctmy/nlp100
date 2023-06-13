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
import optuna

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
        self.embedding = nn.Embedding.from_pretrained(emb_weights)
        self.bilstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, num_layers=2, bidirectional=True)  # 双方向，2層
        self.label = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedding = self.embedding(x)
        out, hidden = self.bilstm(embedding, None)
        out = self.dropout(out[:, -1])
        # outは(batch_size，embedding_size, hidden_size), hiddenは使わない
        return self.label(out)


def collate_fn(batch):
    x, y = list(zip(*batch))
    x = list(rnn.pad_sequence(x, batch_first=True))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


train_data = CustomDataset("../chapter8/train.txt")
test_data = CustomDataset("../chapter8/test.txt")

# GPUにする
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 10
loss_fn = nn.CrossEntropyLoss()


def objective(trial):
    batch_size = int(trial.suggest_int("batch_size", 1, 512, log=True))  # 間隔は4
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)  # 対数連続値
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RAdam"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTM(300, 4, 50, weights, dropout)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # マルチGPUになるように
    model.to(device)
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    minloss = 10.0
    for epoch in range(epochs):
        # 学習
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            # 予測と損失の計算
            pred = model(X)
            loss = loss_fn(pred, y)
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()  # 勾配を計算
            optimizer.step()  # パラメータの最適化
        # 評価
        model.eval()
        size = len(test_dataloader.dataset)
        loss = 0
        with torch.no_grad():  # 勾配計算をしない
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss += loss_fn(pred, y).item()  # 損失
            loss = loss / size
        if loss < minloss:
            minloss = loss
    return minloss


study = optuna.create_study()
study.optimize(objective, n_trials=50)

print(study.best_params)
print(f"loss: {study.best_value}")

'''
{'batch_size': 124, 'dropout': 0.27346271235362063, 'optimizer': 'Adam', 'learning_rate': 1.7519464179271924e-06}
loss: 0.010569588355217381
'''
