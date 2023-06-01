import nlp100_70
from torch import nn
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time

train_data = nlp100_70.CustomDataset("train.txt")
valid_data = nlp100_70.CustomDataset("valid.txt")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=True)

# GPUにする
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0)  # 重みを平均0，分散1の正規分布の乱数に

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()
model = torch.nn.DataParallel(model, device_ids=[0, 1])  # マルチGPUになるように
model.to(device)

learning_rate = 1e-3
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.module.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータの最適化


def accuracyandloss(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, correct = 0, 0
    with torch.no_grad():  # 勾配計算をしない
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()  # 損失
            correct += (torch.argmax(pred, dim=1) == y).sum().item()  # predとyの一致する要素の数
        accuracy = correct / size
        loss = loss / size
    print(f"loss:{loss} accuracy:{accuracy}")
    return [loss, accuracy]


log_train = []
log_valid = []
plt.figure()
with open("time.txt", mode="w")as f:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print(f"\nEpoch {t+1}, B={2**t}\n-------------------------------", file=f)
        start = time.time()
        train_loop(DataLoader(train_data, batch_size=2**t, shuffle=True), model, loss_fn, optimizer)
        end = time.time()
        log_train.append(accuracyandloss(train_dataloader, model, loss_fn))
        log_valid.append(accuracyandloss(valid_dataloader, model, loss_fn))
        g1 = plt.subplot(121)
        g1.plot(np.array(log_train).T[0], color='C0', label='train')
        g1.plot(np.array(log_valid).T[0], color='C1', label='valid')
        g1.set_xlabel('epoch')
        g1.set_ylabel('loss')
        g1.legend()
        g2 = plt.subplot(122)
        g2.plot(np.array(log_train).T[1], color='C0', label='train')
        g2.plot(np.array(log_valid).T[1], color='C1', label='valid')
        g2.set_xlabel('epoch')
        g2.set_ylabel('accuracy')
        g2.legend()
        plt.pause(0.001)  # リアルタイムで描画
        # plt.savefig("plot.png")
        g1.remove()
        g2.remove()
        print(f"time: {end-start}", file=f)
print("Done!")
