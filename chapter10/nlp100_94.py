import nlp100_90
import json
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import queue
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import matplotlib.pyplot as plt


with open("word_id_dict_en.json")as f:
    word_id_dict_en = json.load(f)
with open("word_id_dict_ja.json")as f:
    word_id_dict_ja = json.load(f)


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

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings


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


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


PAD_IDX = 1


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_VOCAB_SIZE = len(word_id_dict_ja)
TGT_VOCAB_SIZE = len(word_id_dict_en)
EMB_SIZE = 100  # embeddingの次元
NHEAD = 4  # マルチヘッドアテンションのヘッドの数
FFN_HID_DIM = 100  # エンコーダーのフィードフォワードの次元
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
model.load_state_dict(torch.load("transformer_weights.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model.to(device)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys


class BeamSearchNode:
    def __init__(self, sequence, log_prob_sum, length):
        self.sequence = sequence
        self.log_prob_sum = log_prob_sum
        self.length = length

    def __lt__(self, other):  # 対数生成確率の平均
        return self.log_prob_sum / self.length > other.log_prob_sum / other.length


def beam_decode(model, src, src_mask, max_len, start_symbol, beam_width=5):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)

    initial_node = BeamSearchNode(
        sequence=torch.tensor([[start_symbol]], dtype=torch.long).to(device),
        log_prob_sum=0.0,
        length=1
    )

    beam = queue.PriorityQueue()
    beam.put(initial_node)
    finished_hypotheses = []

    while not beam.empty() and len(finished_hypotheses) < beam_width:
        node = beam.get()
        ys = node.sequence
        log_prob_sum = node.log_prob_sum
        length = node.length

        if length >= max_len:
            finished_hypotheses.append(node)
            continue

        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])

        top_prob, top_idx = torch.topk(prob, k=beam_width)

        for i in range(beam_width):
            next_word = top_idx[0][i].item()
            prob_word = top_prob[0][i].item()
            new_log_prob_sum = log_prob_sum + torch.log(torch.tensor(prob_word))
            new_ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

            new_node = BeamSearchNode(
                sequence=new_ys,
                log_prob_sum=new_log_prob_sum,
                length=length + 1
            )

            if next_word == 3:  # 終了トークン
                finished_hypotheses.append(new_node)
            else:
                beam.put(new_node)

    finished_hypotheses.sort(key=lambda x: x.log_prob_sum / x.length, reverse=True)

    best_sequence = finished_hypotheses[0].sequence
    return best_sequence


def translate(model, src_sentence, width):
    model.eval()
    src = torch.tensor(nlp100_90.sequence2id(src_sentence, word_id_dict_ja)).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=2, beam_width=width).flatten()
    # result = nlp100_90.id2seqense(tgt_tokens, word_id_dict_en).replace("<start>", "").replace("<end>", "")
    return tgt_tokens


if __name__ == "__main__":
    with open('kftt-data-1.0/data/tok/kyoto-dev.ja') as f1, open('kftt-data-1.0/data/tok/kyoto-dev.en') as f2:
        ja = [line.rstrip() for line in f1.readlines()]
        en = [line.rstrip() for line in f2.readlines()]

    scores = []
    prm = [1, 3, 5, 10, 20]
    for j in prm:
        bleu = BLEU(effective_order=True)  # Trueにすることでマッチするn-gramの順序が連続していない場合にも、より高いスコアが与えられるらしい
        bleuscore = 0
        for i in tqdm(range(len(ja))):
            bleuscore += bleu.sentence_score(str(translate(model, ja[i], j)), [str(nlp100_90.sequence2id(en[i], word_id_dict_en))]).score
        scores.append(bleuscore)
        print(f"BLEU SCORE: {bleuscore/len(ja)}")

    plt.plot(prm, scores)
    plt.ylim(0, 30)
    plt.ylabel('BLEU')
    plt.xlabel('beam_width')
    plt.savefig("ans94.png")
    plt.show()


'''
Using cuda device
100%|██████████████████████████████████████████████████████████| 1166/1166 [03:22<00:00,  5.77it/s]
BLEU SCORE: 24.252294678060437
100%|██████████████████████████████████████████████████████████| 1166/1166 [30:43<00:00,  1.58s/it]
BLEU SCORE: 22.212393863931627
100%|██████████████████████████████████████████████████████████| 1166/1166 [21:50<00:00,  1.12s/it]
BLEU SCORE: 21.264526181326204
100%|██████████████████████████████████████████████████████████| 1166/1166 [19:12<00:00,  1.01it/s]
BLEU SCORE: 20.48943736909106
100%|██████████████████████████████████████████████████████████| 1166/1166 [34:42<00:00,  1.79s/it]
BLEU SCORE: 20.06812449616632
'''
