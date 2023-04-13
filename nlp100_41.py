class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []


class Morph:
    def __init__(self, line):  # 1行を受け取ってMorphオブジェクトにする
        morph = line.split('\t')
        morph[1] = morph[1].split(',')
        self.surface = morph[0]
        self.base = morph[1][6]
        self.pos = morph[1][0]
        self.pos1 = morph[1][1]


def load_result():
    sentences = []
    chunks = []  # 一文のChunkオブジェクトのリスト
    morphs = []  # 一文の係り受け解析結果のリスト
    dst = None
    with open('ai.ja.txt.parsed', mode='r') as f:
        for line in f:
            if line[0] == '*':  # ＊で始まる行なら係先インデックス番号を取得し、一つ前の文節をchunksリストに登録
                if len(morphs) > 0:
                    chunks.append(Chunk(morphs, dst))
                    morphs = []
                dst = int(line.split(' ')[2].rstrip('D'))
            elif line != 'EOS\n':  # 文末以外なら係り受け解析結果をmorphsに追加
                morphs.append(Morph(line))
            else:  # 文末なら文節をchunksリストに登録し、係もとインデックス番号を登録、1文ごとの情報をまとめる
                chunks.append(Chunk(morphs, dst))
                for i, chunk in enumerate(chunks):
                    if chunk.dst != -1:
                        chunks[chunk.dst].srcs.append(i)
                sentences.append(chunks)
                morphs = []
                chunks = []
    return sentences


if __name__ == "__main__":
    sentences = load_result()
    for chunks in sentences:
        for chunk in chunks:
            print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)


# 4.7コードを段落に分割する,p.51

'''
長かったため最後の方だけ

['確か', 'でしょ', 'う', '」', 'と'] 24 [22]
['し', 'て', 'いる', '。'] 34 [0, 14, 23]
['伊勢田', 'は', '、'] 34 []
['「', '思っ', 'た'] 27 []
['以上', 'に'] 32 [26]
['物理', '学者', 'と'] 29 []
['哲学', '者', 'の'] 30 [28]
['もの', 'の'] 31 [29]
['見え', '方', 'の'] 32 [30]
['違い', 'という', 'の', 'は'] 33 [27, 31]
['大きい', 'の', 'かも', 'しれ', 'ませ', 'ん', '」', 'と'] 34 [32]
['述べ', 'て', 'いる', '。'] -1 [24, 25, 33]
[] -1 []
['対談', 'で'] 20 []
['須藤', 'は'] 16 []
['「', 'これ', 'まで'] 6 []
['けっこう'] 6 []
['長時間'] 6 []
['議論', 'を'] 6 []
['行っ', 'て', 'き', 'まし', 'た', '。'] 15 [2, 3, 4, 5]
['おかげ', 'で', '、'] 10 []
['意見', 'の'] 9 []
['違い', 'は'] 10 [8]
['明らか', 'に', 'なっ', 'た', 'と', 'は'] 11 [7, 9]
['思い', 'ます', 'が', '、'] 15 [10]
['果たして'] 15 []
['何', 'か'] 15 []
['決着', 'が'] 15 []
['つく', 'の', 'でしょ', 'う', 'か', '？', '」', 'と'] 16 [6, 11, 12, 13, 14]
['発言', 'し', '、'] 20 [1, 15]
['伊勢田', 'は'] 20 []
['「', '決着', 'は'] 19 []
['つか', 'ない', 'でしょ', 'う', 'ね', '」', 'と'] 20 [18]
['答え', 'て', 'いる', '。'] -1 [0, 16, 17, 19]
'''
