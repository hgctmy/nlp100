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


def collate_fn(batch):
    X, y = list(zip(*batch))
    x_len = torch.tensor([len(x) for x in X])
    X = list(rnn.pad_sequence(X, batch_first=True))
    X = torch.stack(X)
    y = torch.stack(y)
    return X, y, x_len


train_data = CustomDataset("../chapter8/train.txt")
test_data = CustomDataset("../chapter8/test.txt")

# GPUにする
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

epochs = 10
loss_fn = nn.CrossEntropyLoss()


def objective(trial):
    batch_size = trial.suggest_int("batch_size", 1, 128, log=True)
    # dropout = trial.suggest_float("dropout", 0.1, 0.5)  # 連続値
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RAdam"])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = LSTM(300, 4, 50, weights, 0)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])  # マルチGPUになるように
    model.to(device)
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    minloss = 10.0
    for epoch in range(epochs):
        # 学習
        model.train()
        for X, y, x_len in train_dataloader:
            X, y, x_len = X.to(device), y.to(device), x_len.to(device)
            # 予測と損失の計算
            pred = model(X, x_len)
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
            for X, y, x_len in test_dataloader:
                X, y, x_len = X.to(device), y.to(device), x_len.to(device)
                pred = model(X, x_len)
                loss += loss_fn(pred, y).item()  # 損失
            loss = loss / size
        if loss < minloss:
            minloss = loss
    return minloss


study = optuna.create_study()
study.optimize(objective, n_trials=10)

print(study.best_params)
print(study.best_value)

'''
Using cuda device
[I 2023-06-23 19:52:05,388] A new study created in memory with name: no-name-daebb36d-49ae-4eca-ae0f-11ddc1c33f78
[I 2023-06-23 20:12:45,451] Trial 0 finished with value: 0.8253424388623197 and parameters: {'batch_size': 1, 'optimizer': 'RAdam', 'learning_rate': 1.648216773641837e-05}. Best is trial 0 with value: 0.8253424388623197.
[I 2023-06-23 20:14:42,146] Trial 1 finished with value: 0.0491630032189425 and parameters: {'batch_size': 19, 'optimizer': 'Adam', 'learning_rate': 2.9853972094447232e-05}. Best is trial 1 with value: 0.0491630032189425.
[I 2023-06-23 20:25:31,136] Trial 2 finished with value: 0.1964739541998943 and parameters: {'batch_size': 3, 'optimizer': 'Adam', 'learning_rate': 0.0005105891166554151}. Best is trial 1 with value: 0.0491630032189425.
[I 2023-06-23 20:27:51,962] Trial 3 finished with value: 0.04113946559055634 and parameters: {'batch_size': 17, 'optimizer': 'RAdam', 'learning_rate': 0.0002470464898493579}. Best is trial 3 with value: 0.04113946559055634.
[I 2023-06-23 20:29:00,883] Trial 4 finished with value: 0.03533131277364591 and parameters: {'batch_size': 39, 'optimizer': 'RAdam', 'learning_rate': 2.563570937849835e-06}. Best is trial 4 with value: 0.03533131277364591.
[I 2023-06-23 20:32:31,264] Trial 5 finished with value: 0.05752696006976325 and parameters: {'batch_size': 11, 'optimizer': 'RAdam', 'learning_rate': 0.00043633787311915344}. Best is trial 4 with value: 0.03533131277364591.
[I 2023-06-23 20:50:07,744] Trial 6 finished with value: 1.0869961628268683 and parameters: {'batch_size': 1, 'optimizer': 'AdamW', 'learning_rate': 3.698398139950563e-06}. Best is trial 4 with value: 0.03533131277364591.
[I 2023-06-23 20:52:10,433] Trial 7 finished with value: 0.033831992260609074 and parameters: {'batch_size': 18, 'optimizer': 'AdamW', 'learning_rate': 0.0009389282750946757}. Best is trial 7 with value: 0.033831992260609074.
[I 2023-06-23 20:54:06,430] Trial 8 finished with value: 0.03207747635991498 and parameters: {'batch_size': 19, 'optimizer': 'AdamW', 'learning_rate': 0.0005907947020743106}. Best is trial 8 with value: 0.03207747635991498.
[I 2023-06-23 21:09:59,373] Trial 9 finished with value: 0.283968513570459 and parameters: {'batch_size': 2, 'optimizer': 'Adam', 'learning_rate': 0.00040631676984624}. Best is trial 8 with value: 0.03207747635991498.
{'batch_size': 19, 'optimizer': 'AdamW', 'learning_rate': 0.0005907947020743106}
0.03207747635991498

チューニング前
test_loss: 0.010266701119771782 test_accuracy: 0.7608695652173914
チューニング後 100エポック
test_loss: 0.01856364604683294 test_accuracy: 0.7983508245877061
'''
