import nlp100_86
import nlp100_81
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.utils.rnn as rnn


model = nlp100_86.CNN(300, 4, 50)
learning_rate = 1e-3
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def collate_fn(batch):
    x, y = list(zip(*batch))
    x = list(rnn.pad_sequence(x, batch_first=True))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


train_data = nlp100_81.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/train.txt")
test_data = nlp100_81.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/test.txt")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)


def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
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
            pred = model(X)
            loss += loss_fn(pred, y).item()  # 損失
            correct += (torch.argmax(pred, dim=1) == y).sum().item()  # predとyの一致する要素の数
        accuracy = correct / size
        loss = loss / size
    print(loss, accuracy)


for t in range(epochs):
    print(f"Epoch {t+1}-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accuracyandloss(test_dataloader, model, loss_fn)
print("Done!")

'''
Epoch 1-------------------------------
0.01794238944818591 0.49775112443778113
Epoch 2-------------------------------
0.017566686672070574 0.5464767616191905
Epoch 3-------------------------------
0.01722460408796971 0.5809595202398801
Epoch 4-------------------------------
0.01688473536514271 0.6064467766116941
Epoch 5-------------------------------
0.016553419283304973 0.6274362818590704
Epoch 6-------------------------------
0.016237816219029578 0.636431784107946
Epoch 7-------------------------------
0.015946242181972405 0.6499250374812594
Epoch 8-------------------------------
0.015675189747088317 0.6611694152923538
Epoch 9-------------------------------
0.015426194069088846 0.6686656671664168
Epoch 10-------------------------------
0.015196451182844399 0.6724137931034483
Done!
'''
