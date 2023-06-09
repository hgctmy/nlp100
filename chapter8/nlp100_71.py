import nlp100_70
from torch import nn
from torch.utils.data import DataLoader

# ４つ目までのデータをとってくる
feature, label = next(iter(DataLoader(nlp100_70.CustomDataset("/Users/higuchitomoya/python/nlp100/chapter6/train.txt"), batch_size=4, shuffle=False)))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0)  # 重みを平均0，分散1の正規分布の乱数に

    def forward(self, x):
        logits = self.linear(x)
        return logits


model = NeuralNetwork()
print("y_1:", nn.Softmax(dim=1)(model(feature[:1])))  # dim=1 行ごとの和が1になるように
print("Y:", nn.Softmax(dim=1)(model(feature)))

'''
y_1: tensor([[0.0037, 0.5344, 0.4301, 0.0318]], grad_fn=<SoftmaxBackward>)
Y: tensor([[0.0037, 0.5344, 0.4301, 0.0318],
        [0.1820, 0.1107, 0.6761, 0.0312],
        [0.2252, 0.0704, 0.1040, 0.6004],
        [0.0482, 0.5496, 0.2597, 0.1425]], grad_fn=<SoftmaxBackward>)
'''
