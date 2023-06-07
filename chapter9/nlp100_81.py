import nlp100_80
import json
from torch import nn
import torch
from torch.utils.data import Dataset
import pandas as pd

with open("word_id_dict.json")as f:
    word_id_dict = json.load(f)


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
    def __init__(self, embedding_length, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(len(set(word_id_dict.values())) + 1, embedding_length)  # (単語id数+1,埋め込み次元数)
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedding = self.embedding(x)
        out, hidden = self.lstm(embedding, None)
        # outは(batch_size，embedding_size, hidden_size), hiddenは使わない
        return self.label(out[:, -1])


if __name__ == "__main__":
    model = LSTM(300, 4, 30)
    test, answer = CustomDataset("../chapter6/test.txt").__getitem__(0)
    print(nn.Softmax(dim=1)(model(test.unsqueeze(0))))

'''
tensor([[0.3070, 0.2378, 0.2020, 0.2532]], grad_fn=<SoftmaxBackward>)
'''
