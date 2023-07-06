import nlp100_90
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn
import torch.nn as nn
from torch.nn import Transformer
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
    x = rnn.pad_sequence(x, padding_value=PAD_IDX)
    y = rnn.pad_sequence(y, padding_value=PAD_IDX)

    return x, y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=128, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def generate_square_subsequent_mask(sz):  # 右上三角が-inf,ほかは0
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(train_dataloader, model, optimizer):
    model.train()
    losses = 0
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()

    print(f"train loss: {losses / len(list(train_dataloader))}")


def evaluate(test_dataloader, model):
    model.eval()
    losses = 0
    with torch.no_grad():  # 勾配計算をしない
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

    print(f"test loss: {losses / len(list(test_dataloader))}")


if __name__ == "__main__":
    train_data = CustomDataset("kftt-data-1.0/data/tok/kyoto-train.cln.ja", "kftt-data-1.0/data/tok/kyoto-train.cln.en")
    test_data = CustomDataset("kftt-data-1.0/data/tok/kyoto-test.ja", "kftt-data-1.0/data/tok/kyoto-test.en")
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    SRC_VOCAB_SIZE = len(word_id_dict_ja)
    TGT_VOCAB_SIZE = len(word_id_dict_en)
    EMB_SIZE = 100  # embeddingの次元
    NHEAD = 4  # マルチヘッドアテンションのヘッドの数
    FFN_HID_DIM = 100  # エンコーダーのフィードフォワードの次元
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    # model = torch.nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=1)
    lr = 1e-4  # 学習率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 30

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model, optimizer)
        evaluate(test_dataloader, model)
    print("Done!")

    torch.save(model.state_dict(), 'transformer_weights.pth')
'''
Epoch 30
-------------------------------
train loss: 3.5891330358718205
test loss: 4.30319052773553
'''
