import nlp100_70
from torch import nn
from torch.utils.data import DataLoader

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

# クロスエントロピー損失を損失関数とする
# x_1
loss_fn = nn.CrossEntropyLoss()  # LogSoftmaxとNLLLossを結合した損失関数
loss = loss_fn(model(feature[:1]), label[:1])  # 予測の損失を計算
model.zero_grad()  # 勾配を初期化（蓄積されてしまうため）
loss.backward()  # 勾配を計算
print(f"損失:{loss}")
print(f"勾配:{model.linear.weight.grad}")  # 重みの勾配

# x_1 ~ x_4
loss = loss_fn(model(feature[:4]), label[:4])  # 予測の損失を計算
model.zero_grad()  # 勾配を初期化（蓄積されてしまうため）
loss.backward()  # 勾配を計算
print(f"損失:{loss}")
print(f"勾配:{model.linear.weight.grad}")

'''
損失:3.2643027305603027
勾配:tensor([[-0.0037,  0.0050, -0.0084,  ..., -0.0163, -0.0088,  0.0053],
        [-0.0308,  0.0412, -0.0694,  ..., -0.1349, -0.0725,  0.0440],
        [ 0.0416, -0.0556,  0.0937,  ...,  0.1821,  0.0978, -0.0595],
        [-0.0071,  0.0094, -0.0159,  ..., -0.0309, -0.0166,  0.0101]])
損失:1.680577278137207
勾配:tensor([[-0.0042, -0.0087, -0.0062,  ...,  0.0045, -0.0031,  0.0091],
        [ 0.0119,  0.0246, -0.0041,  ..., -0.0568, -0.0018,  0.0015],
        [-0.0077, -0.0204,  0.0127,  ...,  0.0626,  0.0076, -0.0116],
        [ 0.0001,  0.0044, -0.0025,  ..., -0.0103, -0.0028,  0.0010]])
'''
