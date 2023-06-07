import nlp100_81
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn


model = nlp100_81.LSTM(300, 4, 30)
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
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)


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
Epoch 1
-------------------------------
1.149905288728817 0.48425787106446777
Epoch 2
-------------------------------
1.1135953025811391 0.5342953523238381
Epoch 3
-------------------------------
1.0656155750596674 0.5791791604197901
Epoch 4
-------------------------------
1.004632517578139 0.6233133433283359
Epoch 5
-------------------------------
0.9386735393929629 0.6562968515742129
Epoch 6
-------------------------------
0.8721510096236008 0.6865629685157422
Epoch 7
-------------------------------
0.8060248026684013 0.716547976011994
Epoch 8
-------------------------------
0.7405609799298699 0.7384745127436282
Epoch 9
-------------------------------
0.6765229913552335 0.7603073463268366
Epoch 10
-------------------------------
0.6149462376556442 0.7801724137931034
Done!
'''
