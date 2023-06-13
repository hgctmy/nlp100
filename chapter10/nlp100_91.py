import nlp100_90
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math

with open("word_id_dict_en.json")as f:
    word_id_dict_en = json.load(f)
with open("word_id_dict_ja.json")as f:
    word_id_dict_ja = json.load(f)


class CustomDataset(Dataset):
    def __init__(self, file_path_ja, file_path_en):
        with open(file_path_en)as f1, open(file_path_ja)as f2:
            self.ja = [line.rstrip() for line in f2.readlines()]
            self.en = [line.rstrip() for line in f1.readlines()]

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        ja = torch.tensor(nlp100_90.sequence2id(self.ja[idx], word_id_dict_ja))
        en = torch.tensor(nlp100_90.sequence2id(self.en[idx], word_id_dict_en))
        return ja, en


PAD_IDX = 1


def collate_fn(batch):
    x, y = list(zip(*batch))
    x = list(rnn.pad_sequence(x, padding_value=PAD_IDX))
    y = list(rnn.pad_sequence(y, padding_value=PAD_IDX))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


train_data = CustomDataset("kftt-data-1.0/data/tok/kyoto-train.cln.ja", "kftt-data-1.0/data/tok/kyoto-train.cln.en")
test_data = CustomDataset("kftt-data-1.0/data/tok/kyoto-test.ja", "kftt-data-1.0/data/tok/kyoto-test.en")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)


class TransformerModel(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, embedding_size, vocab_size_src, vocab_size_tgt, dim_feedforward, dropout, nhead):
        super().__init__()
        self.token_embedding_src = TokenEmbedding(vocab_size_src, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.token_embedding_tgt = TokenEmbedding(vocab_size_tgt, embedding_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output = nn.Linear(embedding_size, vocab_size_tgt)

    def forward(self, src, tgt, mask_src, mask_tgt, padding_mask_src, padding_mask_tgt, memory_key_padding_mask):
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src, padding_mask_src)
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(
            embedding_tgt, memory, mask_tgt, None,
            padding_mask_tgt, memory_key_padding_mask
        )
        return self.output(outs)

    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])


def create_mask(src, tgt, PAD_IDX):
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]
    mask_tgt = generate_square_subsequent_mask(seq_len_tgt)
    mask_src = torch.zeros((seq_len_src, seq_len_src), device=device).type(torch.bool)
    padding_mask_src = (src == PAD_IDX).transpose(0, 1)
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1)
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
vocab_size_srcs = len(word_id_dict_ja)
vocab_size_tgt = len(word_id_dict_en)
emsize = 200  # embeddingの次元
nhid = 200  # エンコーダーのフィードフォワードの次元
n_encoder_layers = 2
n_decoder_layers = 2
nhead = 2  # マルチヘッドアテンションのヘッドの数
dropout = 0.2
model = TransformerModel(n_encoder_layers, n_decoder_layers, emsize, vocab_size_srcs, vocab_size_tgt, nhid, dropout, nhead)
model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)


loss_fn = nn.CrossEntropyLoss(ignore_index=1)
lr = 5.0  # 学習率
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
epochs = 3


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        input_y = y[:, -1, :]
        mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(x, input_y, PAD_IDX)
        # 予測と損失の計算
        pred = model(
            src=x, tgt=input_y,
            mask_src=mask_src, mask_tgt=mask_tgt,
            padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
            memory_key_padding_mask=padding_mask_src
        )
        output_y = y[1, :, :]
        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), output_y.reshape(-1))
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()  # 勾配を計算
        optimizer.step()  # パラメータの最適化
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loss(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    loss = 0
    with torch.no_grad():  # 勾配計算をしない
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            input_y = y[:, -1, :]
            mask_src, mask_tgt, padding_mask_src, padding_mask_tgt = create_mask(x, input_y, PAD_IDX)
        # 予測と損失の計算
            pred = model(
                src=x, tgt=input_y,
                mask_src=mask_src, mask_tgt=mask_tgt,
                padding_mask_src=padding_mask_src, padding_mask_tgt=padding_mask_tgt,
                memory_key_padding_mask=padding_mask_src
            )
            output_y = y[1, :, :]
            loss += loss_fn(pred.reshape(-1, pred.shape[-1]), output_y.reshape(-1)).item()  # 損失
        loss = loss / size
    print(f'test_loss: {loss}')


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss(test_dataloader, model, loss_fn)
    scheduler.step()
print("Done!")

torch.save(model.state_dict(), 'transformer_weights.pth')
