import nlp100_81
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn


model = nlp100_81.LSTM(300, 4, 30)
learning_rate = 1e-2
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def collate_fn(batch):
    x, y = list(zip(*batch))
    x = list(rnn.pad_sequence(x, batch_first=True))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


train_data = nlp100_81.CustomDataset("../chapter8/train.txt")
test_data = nlp100_81.CustomDataset("../chapter8/test.txt")
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
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 予測と損失の計算
        pred = model(X)
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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
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
Epoch 100
-------------------------------
loss: 0.067949  [    0/10672]
loss: 0.187536  [ 3200/10672]
loss: 0.080748  [ 6400/10672]
loss: 0.222249  [ 9600/10672]
test_loss: 0.016319330485685655 test_accuracy: 0.7728635682158921
'''
