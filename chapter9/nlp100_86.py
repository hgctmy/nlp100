import nlp100_81
from torch import nn
from torch.nn import functional as F
import json

with open("word_id_dict.json")as f:
    word_id_dict = json.load(f)


class CNN(nn.Module):
    def __init__(self, embedding_length, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(len(set(word_id_dict.values())) + 1, embedding_length)  # (単語id数+1,埋め込み次元数)
        self.conv = nn.Conv1d(embedding_length, hidden_size, 3, padding=1)  # フィルターサイズ3
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, titleid):
        x = self.embedding(titleid)
        x = self.conv(x.transpose(-1, -2))
        x = self.relu(x)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)  # 次元を削除
        x = self.linear(x)
        # outは(batch_size，embedding_size, hidden_size), hiddenは使わない
        return x


if __name__ == "__main__":
    model = CNN(300, 4, 50)
    test, answer = nlp100_81.CustomDataset("../chapter6/test.txt").__getitem__(0)
    print(nn.Softmax(dim=1)(model(test.unsqueeze(0))))

'''
tensor([[0.2734, 0.1332, 0.1988, 0.3945]], grad_fn=<SoftmaxBackward>)
'''
