import nlp100_70
from torch import nn
import torch
from torch.utils.data import DataLoader

train_dataloader = DataLoader(nlp100_70.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/train.txt"), batch_size=64, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0)  # 重みを平均0，分散1の正規分布の乱数に

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()
learning_rate = 1e-3
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータの最適化


for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Done!")

torch.save(model.state_dict(), 'model_weights.pth')
