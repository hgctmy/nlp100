import sentencepiece as spm
import re

# 日本語
spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')

for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)  # 連続するスペースを1個に
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)

# 英語
spm.SentencePieceTrainer.Train('--input=kftt-data-1.0/data/orig/kyoto-train.en --model_prefix=kyoto_en --vocab_size=16000 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()
sp.Load('kyoto_en.model')

for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.en', 'train.sub.en'),
    ('kftt-data-1.0/data/orig/kyoto-dev.en', 'dev.sub.en'),
    ('kftt-data-1.0/data/orig/kyoto-test.en', 'test.sub.en'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)  # 連続するスペースを1個に
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)
