from nltk.translate.bleu_score import sentence_bleu
import nlp100_91
from torch.utils.data import DataLoader
import json
import torch


with open("word_id_dict_en.json")as f:
    word_id_dict_en = json.load(f)
with open("word_id_dict_ja.json")as f:
    word_id_dict_ja = json.load(f)


SRC_VOCAB_SIZE = len(word_id_dict_ja)
TGT_VOCAB_SIZE = len(word_id_dict_en)
EMB_SIZE = 100  # embeddingの次元
NHEAD = 4  # マルチヘッドアテンションのヘッドの数
FFN_HID_DIM = 100  # エンコーダーのフィードフォワードの次元
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


model = nlp100_91.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
model.load_state_dict(torch.load("transformer_weights.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model.to(device)
test_data = nlp100_91.CustomDataset("kftt-data-1.0/data/tok/kyoto-test.ja", "kftt-data-1.0/data/tok/kyoto-test.en")
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=nlp100_91.collate_fn)


def evaluate(test_dataloader, model):
    model.eval()
    size = len(test_dataloader.dataset)
    bleu = 0
    with torch.no_grad():  # 勾配計算をしない
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = nlp100_91.create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            result = torch.argmax(logits, dim=2)
            for i in range(len(result)):  # 原文と翻訳文の1セットごとに計算し，サイズで割る
                bleu += sentence_bleu([str(j) for j in result[i].tolist()], [str(j) for j in tgt_out[i].tolist()])
        print(f"bleu score: {bleu/size}")


evaluate(test_dataloader, model)

'''
The hypothesis contains 0 counts of 4-gram overlaps.
Therefore the BLEU score evaluates to 0, independently of
how many N-gram overlaps of lower order it contains.
Consider using lower n-gram order or use SmoothingFunction()
  warnings.warn(_msg)
bleu score: 0.00013246643518002648
'''
