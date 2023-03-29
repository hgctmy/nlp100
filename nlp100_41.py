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
