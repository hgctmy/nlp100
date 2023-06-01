import nlp100_70
from torch import nn
import torch
from torch.utils.data import DataLoader

train_data = nlp100_70.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/train.txt")
test_data = nlp100_70.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/test.txt")
train_dataloader = DataLoader(train_data, batch_size=train_data.__len__(), shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0)  # 重みを平均0，分散1の正規分布の乱数に

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()
model.load_state_dict(torch.load('/Users/higuchitomoya/python/nlp100/chapter8/model_weights.pth'))
model.eval()


def accuracy(dataloader, model):
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():  # 勾配計算をしない
        for X, y in dataloader:
            pred = torch.argmax(model(X), dim=1)
            correct += (pred == y).sum().item()  # predとyの一致する要素の数
    print(correct / size)


accuracy(train_dataloader, model)
accuracy(test_dataloader, model)

'''
0.5692466266866567
0.5719640179910045
'''
